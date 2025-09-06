# main.py — NFL Classic Optimizer Backend (FastAPI + OR-Tools)
# ------------------------------------------------------------
# Requirements:
#   pip install fastapi uvicorn "pydantic>=2" ortools
#
# Endpoints:
#   POST /solve_nfl_stream  (SSE; streams one lineup per event)
#   POST /solve_nfl         (batch; returns all lineups in one JSON)

from typing import List, Dict, Optional, Literal, Tuple, Set
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from starlette.responses import StreamingResponse
from ortools.sat.python import cp_model
import random
import json
import logging
import traceback

# ----------------------------- app & CORS -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten for prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger("optimizer")
logging.basicConfig(level=logging.INFO)

# Streaming headers (work well with Nginx/Cloudflare/Render)
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# ----------------------------- SSE helper -----------------------------
def sse_event(payload: dict) -> bytes:
    return f"event: progress\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

# ----------------------------- NFL models -----------------------------
class NFLPlayer(BaseModel):
    name: str
    pos: Literal["QB", "RB", "WR", "TE", "DST"]
    team: str
    opp: str
    salary: int
    proj: float
    floor: float
    ceil: float
    pown: float   # 0..1
    opt: float    # 0..1

class Slot(BaseModel):
    name: str
    eligible: List[Literal["QB","RB","WR","TE","DST"]]

class GroupRule(BaseModel):
    mode: Literal["at_most", "at_least", "exactly"] = "at_most"
    count: int = 1
    players: List[str] = Field(default_factory=list)

class TeamStackRule(BaseModel):
    team: str
    qb_stack_min: Optional[int] = None
    bringback_min: Optional[int] = None
    allow_rb_in_stack: Optional[bool] = None
    bringback_teams: Optional[List[str]] = None
    max_from_team: Optional[int] = None

class SolveNFLRequest(BaseModel):
    site: Literal["dk","fd"]
    slots: List[Slot]
    players: List[NFLPlayer]
    n: int
    cap: int
    objective: Literal["proj","floor","ceil","pown","opt"] = "proj"

    locks: List[str] = Field(default_factory=list)
    excludes: List[str] = Field(default_factory=list)
    boosts: Dict[str, int] = Field(default_factory=dict)
    randomness: float = 0.0
    global_max_pct: float = 100.0
    min_pct: Dict[str, float] = Field(default_factory=dict)
    max_pct: Dict[str, float] = Field(default_factory=dict)
    min_diff: int = 1
    time_limit_ms: int = 1500

    qb_stack_min: int = 2
    stack_allow_rb: bool = False
    bringback_min: int = 1
    max_from_team: Optional[int] = None
    avoid_rb_vs_opp_dst: bool = True
    avoid_offense_vs_opp_dst: bool = False

    groups: List[GroupRule] = Field(default_factory=list)
    team_stack_rules: List[TeamStackRule] = Field(default_factory=list)

    # QB-team exposure caps across the run (e.g., {"BUF": 40, "KC": 35})
    team_max_pct: Dict[str, float] = Field(default_factory=dict)

    # Cap on SUM of lineup pOWN percentage points (e.g., 200 means 200 total pOWN points)
    lineup_pown_max: Optional[float] = None

    # 🔹 Accept legacy/frontend key too; harmonized in validator below
    max_lineup_pown_pct: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _unify_lineup_pown_cap(cls, values):
        if values.get("lineup_pown_max") is None and values.get("max_lineup_pown_pct") is not None:
            values["lineup_pown_max"] = values.get("max_lineup_pown_pct")
        return values

# ----------------------------- helpers -----------------------------
def _nfl_metric(p: NFLPlayer, objective: str) -> float:
    if objective == "proj": return p.proj
    if objective == "floor": return p.floor
    if objective == "ceil":  return p.ceil
    if objective == "pown":  return p.pown * 100.0
    if objective == "opt":   return p.opt * 100.0
    return p.proj

def _nfl_score(p: NFLPlayer, objective: str, boost_steps: int, randomness_pct: float) -> float:
    base = _nfl_metric(p, objective)
    boosted = base * (1.0 + 0.03 * (boost_steps or 0))
    if randomness_pct > 0:
        r = randomness_pct / 100.0
        boosted *= (1.0 + random.uniform(-r, r))
    return boosted

def _team_to_opp_map(players: List[NFLPlayer]) -> Dict[str, Set[str]]:
    m: Dict[str, Set[str]] = {}
    for p in players:
        if p.team and p.opp:
            m.setdefault(p.team, set()).add(p.opp)
    return m

def _caps_from_pct(N: int, pct: float) -> int:
    return int(max(0.0, min(100.0, pct)) / 100.0 * N)

def _player_caps(req: SolveNFLRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    gcap = _caps_from_pct(N, req.global_max_pct)
    caps: Dict[str, int] = {}
    for p in req.players:
        per = _caps_from_pct(N, req.max_pct.get(p.name, 100.0))
        caps[p.name] = min(per, gcap) if gcap > 0 else 0
    return caps

def _qb_team_caps(req: SolveNFLRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    out: Dict[str, int] = {}
    for team, pct in (req.team_max_pct or {}).items():
        out[team] = _caps_from_pct(N, pct)
    return out

def _min_need_player(req: SolveNFLRequest, counts: Dict[str, int]) -> Optional[str]:
    needs = []
    N = max(1, int(req.n))
    for p in req.players:
        need = _caps_from_pct(N, req.min_pct.get(p.name, 0.0))
        if counts.get(p.name, 0) < need:
            needs.append(p)
    if not needs:
        return None
    needs.sort(key=lambda r: r.proj, reverse=True)
    return needs[0].name

def _scores_for_iteration(req: SolveNFLRequest) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for p in req.players:
        b = (req.boosts or {}).get(p.name, 0)
        scores[p.name] = _nfl_score(p, req.objective, b, req.randomness)
    return scores

def _team_rule_for(team: str, rules: List[TeamStackRule]) -> TeamStackRule:
    for r in rules or []:
        if r.team == team:
            return r
    return TeamStackRule(team=team)

# ----------------------------- core solver -----------------------------
def _solve_one_nfl(
    req: SolveNFLRequest,
    scores: Dict[str, float],
    used_player_counts: Dict[str, int],
    player_caps: Dict[str, int],
    forced_include: Optional[str],
    prior_lineups: List[List[str]],
    qb_team_caps_used: Dict[str, int],
    qb_team_caps: Dict[str, int],
) -> Optional[Tuple[List[str], int, float, str]]:

    model = cp_model.CpModel()

    # convenience maps
    by_name: Dict[str, NFLPlayer] = {p.name: p for p in req.players}
    names = list(by_name.keys())
    slot_names = [s.name for s in req.slots]

    # decision vars: x[name, slot] and helper y[name]
    x: Dict[tuple, cp_model.IntVar] = {}
    for n in names:
        for sl in slot_names:
            x[(n, sl)] = model.NewBoolVar(f"x_{n}_{sl}")

    y: Dict[str, cp_model.IntVar] = {}
    for n in names:
        var = model.NewBoolVar(f"y_{n}")
        y[n] = var
        model.Add(sum(x[(n, sl)] for sl in slot_names) == var)

    # fill each slot with exactly one eligible
    for sl, slot in zip(slot_names, req.slots):
        allowed = set(slot.eligible or [])
        for n in names:
            if by_name[n].pos not in allowed:
                model.Add(x[(n, sl)] == 0)
        model.Add(sum(x[(n, sl)] for n in names) == 1)

    # at most one slot per player
    for n in names:
        model.Add(sum(x[(n, sl)] for sl in slot_names) <= 1)

    # salary cap
    model.Add(sum(y[n] * by_name[n].salary for n in names) <= req.cap)

    # locks / excludes
    excl = set(req.excludes or [])
    lock = set(req.locks or [])
    for n in names:
        if n in excl:
            model.Add(y[n] == 0)
        if n in lock:
            model.Add(y[n] == 1)

    # per-player exposure caps already used across the run
    for n in names:
        if used_player_counts.get(n, 0) >= player_caps.get(n, 10**9):
            model.Add(y[n] == 0)

    # forced include (min%)
    if forced_include and forced_include in y:
        model.Add(y[forced_include] == 1)

    # min_diff vs prior lineups (Hamming distance)
    roster_size = len(req.slots)
    for lineup in prior_lineups:
        model.Add(sum(y[n] for n in lineup if n in y) <= roster_size - max(1, req.min_diff))

    # group rules (by player name)
    for g in req.groups or []:
        group_vars = [y[n] for n in g.players if n in y]
        if not group_vars:
            continue
        s = sum(group_vars)
        if g.mode == "at_most":
            model.Add(s <= g.count)
        elif g.mode == "at_least":
            model.Add(s >= g.count)
        elif g.mode == "exactly":
            model.Add(s == g.count)

    # ---- team-level constraints
    by_team: Dict[str, List[str]] = {}
    for n in names:
        by_team.setdefault(by_name[n].team, []).append(n)

    team_limit: Dict[str, Optional[int]] = {}
    if isinstance(req.max_from_team, int) and req.max_from_team is not None:
        for t in by_team.keys():
            team_limit[t] = req.max_from_team
    for r in (req.team_stack_rules or []):
        if r.team and isinstance(r.max_from_team, int) and r.max_from_team is not None:
            team_limit[r.team] = r.max_from_team

    for t, members in by_team.items():
        lim = team_limit.get(t)
        if isinstance(lim, int) and lim is not None:
            model.Add(sum(y[n] for n in members) <= lim)

    # avoid RB vs opp DST / avoid offense vs opp DST
    offense_pos = {"QB", "RB", "WR", "TE"}
    for n in names:
        p = by_name[n]
        if p.pos == "DST":
            for m in names:
                q = by_name[m]
                if q.team == p.opp:
                    if req.avoid_offense_vs_opp_dst and q.pos in offense_pos:
                        model.Add(y[n] + y[m] <= 1)
                    elif req.avoid_rb_vs_opp_dst and q.pos == "RB":
                        model.Add(y[n] + y[m] <= 1)

    # require exactly 1 QB
    qb_names = [n for n in names if by_name[n].pos == "QB"]
    if qb_names:
        model.Add(sum(y[n] for n in qb_names) == 1)

    # map: team -> players by position
    team_to_players: Dict[str, Dict[str, List[str]]] = {}
    for n in names:
        p = by_name[n]
        team_to_players.setdefault(p.team, {}).setdefault(p.pos, []).append(n)

    team_to_opps = _team_to_opp_map(req.players)

    def effective_rule(team: str):
        r = _team_rule_for(team, req.team_stack_rules)
        qb_stack = r.qb_stack_min if r.qb_stack_min is not None else req.qb_stack_min
        bring_min = r.bringback_min if r.bringback_min is not None else req.bringback_min
        allow_rb = r.allow_rb_in_stack if r.allow_rb_in_stack is not None else req.stack_allow_rb
        bring_teams = r.bringback_teams
        return qb_stack, bring_min, allow_rb, bring_teams

    # QB team indicators
    qb_team_vars: Dict[str, cp_model.IntVar] = {}
    for team in team_to_players.keys():
        v = model.NewBoolVar(f"qb_team_{team}")
        qb_team_vars[team] = v
        team_qbs = [n for n in qb_names if by_name[n].team == team]
        if team_qbs:
            model.Add(sum(y[n] for n in team_qbs) == v)
        else:
            model.Add(v == 0)

    # ---- Enforce QB-team exposure caps across the run
    for team, v in qb_team_vars.items():
        cap = qb_team_caps.get(team, 10**9)
        used = qb_team_caps_used.get(team, 0)
        if used >= cap:
            model.Add(v == 0)

    # ---- Stacking / bringback (hard constraints)
    for team, v in qb_team_vars.items():
        qb_stack, bring_min, allow_rb_in_stack, bring_teams = effective_rule(team)
        helper_pos = {"WR", "TE"}
        if allow_rb_in_stack:
            helper_pos.add("RB")

        helpers_same = [n for n in names if by_name[n].team == team and by_name[n].pos in helper_pos]
        if qb_stack > 0:
            if helpers_same:
                model.Add(sum(y[n] for n in helpers_same) >= qb_stack * v)
            else:
                # Require stacks but no eligible helpers -> forbid this QB team
                model.Add(v == 0)

        opp_teams = set(bring_teams or []) or set(team_to_opps.get(team, set()))
        if bring_min > 0:
            if opp_teams:
                bring_pos = {"WR", "TE"}
                if allow_rb_in_stack:
                    bring_pos.add("RB")
                br_vars = [y[n] for n in names if by_name[n].team in opp_teams and by_name[n].pos in bring_pos]
                if br_vars:
                    model.Add(sum(br_vars) >= bring_min * v)
                else:
                    model.Add(v == 0)
            else:
                model.Add(v == 0)

    # lineup pOWN% cap (percentage points; e.g., 200 means total pOWN sum ≤ 200)
    if req.lineup_pown_max is not None:
        def _safe_pct(x):
            try: return float(x or 0.0)
            except: return 0.0
        cap = int(round(max(0.0, float(req.lineup_pown_max))))
        lhs = sum(int(round(_safe_pct(by_name[n].pown) * 100.0)) * y[n] for n in names)
        model.Add(lhs <= cap)

    # objective
    model.Maximize(sum(int(scores[n] * 1000) * y[n] for n in names))

    # solve
    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = [n for n in names if solver.Value(y[n]) == 1]
    if len(chosen) != len(set(chosen)) or len(chosen) != len(req.slots):
        return None

    salary = sum(by_name[n].salary for n in chosen)
    total = sum(_nfl_metric(by_name[n], req.objective) for n in chosen)

    qb_team = ""
    for team, v in qb_team_vars.items():
        if solver.Value(v) == 1:
            qb_team = team
            break

    return chosen, salary, total, qb_team

# ----------------------------- endpoints -----------------------------
@app.post("/solve_nfl_stream")
def solve_nfl_stream(req: SolveNFLRequest):
    try:
        random.seed()

        used_player_counts: Dict[str, int] = {}
        prior: List[List[str]] = []
        player_caps = _player_caps(req)

        qb_team_caps_used: Dict[str, int] = {}
        qb_team_caps = _qb_team_caps(req)

        def gen():
            # small heartbeat so proxies flush early
            yield b":hb\n\n"
            produced = 0
            for i in range(req.n):
                scores = _scores_for_iteration(req)
                include = _min_need_player(req, used_player_counts)

                ans = _solve_one_nfl(
                    req, scores, used_player_counts, player_caps,
                    include, prior, qb_team_caps_used, qb_team_caps
                )
                if not ans:
                    yield sse_event({"done": True, "reason": "no_more_unique_or_exposure_capped", "produced": produced})
                    return

                chosen, salary, total, qb_team = ans

                for n in chosen:
                    used_player_counts[n] = used_player_counts.get(n, 0) + 1
                if qb_team:
                    qb_team_caps_used[qb_team] = qb_team_caps_used.get(qb_team, 0) + 1

                prior.append(chosen)
                produced += 1

                yield sse_event({
                    "index": i + 1,
                    "drivers": sorted(chosen),
                    "salary": salary,
                    "total": total,
                })

            yield sse_event({"done": True, "produced": produced})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    except Exception as e:
        log.error("solve_nfl_stream error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve_nfl")
def solve_nfl(req: SolveNFLRequest):
    try:
        random.seed()

        used_player_counts: Dict[str, int] = {}
        prior: List[List[str]] = []
        player_caps = _player_caps(req)

        qb_team_caps_used: Dict[str, int] = {}
        qb_team_caps = _qb_team_caps(req)

        out = []
        for _ in range(req.n):
            scores = _scores_for_iteration(req)
            include = _min_need_player(req, used_player_counts)

            ans = _solve_one_nfl(
                req, scores, used_player_counts, player_caps,
                include, prior, qb_team_caps_used, qb_team_caps
            )
            if not ans:
                break

            chosen, salary, total, qb_team = ans

            for n in chosen:
                used_player_counts[n] = used_player_counts.get(n, 0) + 1
            if qb_team:
                qb_team_caps_used[qb_team] = qb_team_caps_used.get(qb_team, 0) + 1

            prior.append(chosen)
            out.append({"drivers": sorted(chosen), "salary": salary, "total": total})

        return {"lineups": out, "produced": len(out)}
    except Exception as e:
        log.error("solve_nfl error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================ SHOWDOWN SOLVER ============================
# Paste this entire block at the very bottom of main.py

from typing import List, Dict, Optional, Literal, Tuple, Set
from pydantic import BaseModel, Field

# ---- Models -------------------------------------------------------------

ShowPos = Literal["QB", "RB", "WR", "TE", "DST", "K"]
ShowCap = Literal["CPT", "MVP", "FLEX"]  # UI uses CPT for DK, MVP for FD

class SDPlayer(BaseModel):
    name: str
    pos: ShowPos
    team: str
    opp: str
    salary: int
    proj: float
    floor: float
    ceil: float
    pown: float   # 0..1
    opt: float    # 0..1
    # Slot-specific values; if omitted, sensible fallbacks are used
    cap_salary: Optional[int] = None
    cap_proj: Optional[float] = None
    cap_floor: Optional[float] = None
    cap_ceil: Optional[float] = None
    cap_pown: Optional[float] = None   # 0..1
    cap_opt: Optional[float] = None    # 0..1

class SDSlot(BaseModel):
    name: ShowCap
    eligible: List[ShowPos]

class SDIfThenRule(BaseModel):
    # IF side
    if_slot: ShowCap = "CPT"  # "CPT"|"MVP"|"FLEX"
    if_pos: List[ShowPos] = Field(default_factory=lambda: ["QB"])
    if_team_exact: Optional[str] = None
    # THEN side
    then_at_least: int = 1
    from_pos: List[ShowPos] = Field(default_factory=lambda: ["WR", "TE"])
    team_scope: Literal["same_team", "opp_team", "any", "exact_team"] = "same_team"
    team_exact: Optional[str] = None

class SDSolveRequest(BaseModel):
    site: Literal["dk","fd"]
    slots: List[SDSlot]
    players: List[SDPlayer]
    n: int
    cap: int
    objective: Literal["proj","floor","ceil","pown","opt"] = "proj"

    locks: List[str] = Field(default_factory=list)
    excludes: List[str] = Field(default_factory=list)
    boosts: Dict[str, int] = Field(default_factory=dict)
    randomness: float = 0.0
    global_max_pct: float = 100.0
    min_pct: Dict[str, float] = Field(default_factory=dict)
    max_pct: Dict[str, float] = Field(default_factory=dict)

    # per-tag exposure controls: keys like "Name::CPT" / "Name::MVP" / "Name::FLEX"
    min_pct_tag: Dict[str, float] = Field(default_factory=dict)
    max_pct_tag: Dict[str, float] = Field(default_factory=dict)

    min_diff: int = 1
    time_limit_ms: int = 1500

    lineup_pown_max: Optional[float] = None
    rules: List[SDIfThenRule] = Field(default_factory=list)

# ---- Helpers ------------------------------------------------------------

def _sd_metric(p: SDPlayer, objective: str) -> float:
    if objective == "proj": return p.proj
    if objective == "floor": return p.floor
    if objective == "ceil":  return p.ceil
    if objective == "pown":  return p.pown * 100.0
    if objective == "opt":   return p.opt * 100.0
    return p.proj

def _sd_score(p: SDPlayer, objective: str, boost_steps: int, randomness_pct: float) -> float:
    base = _sd_metric(p, objective)
    boosted = base * (1.0 + 0.03 * (boost_steps or 0))
    if randomness_pct > 0:
        r = randomness_pct / 100.0
        boosted *= (1.0 + random.uniform(-r, r))
    return boosted

def _sd_salary_for_slot(p: SDPlayer, slot_label: str) -> int:
    if slot_label in ("CPT", "MVP"):
        return int(p.cap_salary if p.cap_salary is not None else round(p.salary * 1.5))
    return p.salary

def _sd_metric_for_slot(p: SDPlayer, slot_label: str, objective: str) -> float:
    if slot_label in ("CPT", "MVP"):
        if objective == "proj":
            return p.cap_proj if p.cap_proj is not None else p.proj * 1.5
        if objective == "floor":
            return p.cap_floor if p.cap_floor is not None else p.floor * 1.5
        if objective == "ceil":
            return p.cap_ceil if p.cap_ceil is not None else p.ceil * 1.5
        if objective == "pown":
            return (p.cap_pown if p.cap_pown is not None else p.pown) * 100.0
        if objective == "opt":
            return (p.cap_opt if p.cap_opt is not None else p.opt) * 100.0
    return _sd_metric(p, objective)

def _sd_caps_from_pct(N: int, pct: float) -> int:
    return int(max(0.0, min(100.0, pct)) / 100.0 * N)

def _sd_player_caps(req: SDSolveRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    gcap = _sd_caps_from_pct(N, req.global_max_pct)
    caps: Dict[str, int] = {}
    for p in req.players:
        per = _sd_caps_from_pct(N, req.max_pct.get(p.name, 100.0))
        caps[p.name] = min(per, gcap) if gcap > 0 else 0
    return caps

def _sd_player_caps_tag(req: SDSolveRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    gcap = _sd_caps_from_pct(N, req.global_max_pct)
    out: Dict[str, int] = {}
    for key, pct in (req.max_pct_tag or {}).items():
        out[key] = min(_sd_caps_from_pct(N, pct), gcap) if gcap > 0 else 0
    return out

def _sd_min_need_player(req: SDSolveRequest, counts: Dict[str, int]) -> Optional[str]:
    needs = []
    N = max(1, int(req.n))
    for p in req.players:
        need = _sd_caps_from_pct(N, req.min_pct.get(p.name, 0.0))
        if counts.get(p.name, 0) < need:
            needs.append(p)
    if not needs:
        return None
    needs.sort(key=lambda r: r.proj, reverse=True)
    return needs[0].name

def _sd_min_need_tag(req: SDSolveRequest, counts_tag: Dict[str, int]) -> Optional[str]:
    N = max(1, int(req.n))
    needs: List[Tuple[str,float]] = []
    for key, pct in (req.min_pct_tag or {}).items():
        need = _sd_caps_from_pct(N, pct)
        if counts_tag.get(key, 0) < need:
            try:
                nm, _sl = key.rsplit("::", 1)
            except ValueError:
                continue
            for p in req.players:
                if p.name == nm:
                    needs.append((key, p.proj))
                    break
    if not needs:
        return None
    needs.sort(key=lambda t: t[1], reverse=True)
    return needs[0][0]

def _sd_team_to_opp_map(players: List[SDPlayer]) -> Dict[str, Set[str]]:
    m: Dict[str, Set[str]] = {}
    for p in players:
        if p.team and p.opp:
            m.setdefault(p.team, set()).add(p.opp)
    return m

def _sd_ordered_players(slot_map: Dict[str, str]) -> List[str]:
    order: List[str] = []
    cap = slot_map.get("CPT") or slot_map.get("MVP")
    if cap:
        order.append(cap)
    i = 1
    while True:
        key = f"FLEX{i}"
        if key in slot_map:
            order.append(slot_map[key])
            i += 1
        else:
            break
    for k in sorted(slot_map.keys()):
        if k not in ("CPT", "MVP") and not k.startswith("FLEX"):
            order.append(slot_map[k])
    return order

def _sd_canonical_label(slot_id: str) -> str:
    """Map engine slot ids to canonical tag: FLEX1/2/... -> FLEX; MVP/CPT -> themselves."""
    s = (slot_id or "").upper()
    return "FLEX" if s.startswith("FLEX") else s

# ---- Core solver --------------------------------------------------------

def _sd_solve_one(
    req: SDSolveRequest,
    scores: Dict[str, float],
    used_player_counts: Dict[str, int],
    player_caps: Dict[str, int],
    forced_include: Optional[str],
    prior_lineups: List[List[str]],
    used_tag_counts: Dict[str, int],
    tag_caps: Dict[str, int],
) -> Optional[Tuple[List[str], int, float, Dict[str, str]]]:
    model = cp_model.CpModel()

    slot_labels = [s.name for s in req.slots]
    slot_ids: List[str] = []
    seen: Dict[str, int] = {}
    for lbl in slot_labels:
        if lbl in ("FLEX",):
            seen[lbl] = seen.get(lbl, 0) + 1
            slot_ids.append(f"{lbl}{seen[lbl]}")
        else:
            slot_ids.append(lbl)
    id_to_label = {sl: lbl for sl, lbl in zip(slot_ids, slot_labels)}

    by_name: Dict[str, SDPlayer] = {p.name: p for p in req.players}
    names = list(by_name.keys())

    x: Dict[Tuple[str,str], cp_model.IntVar] = {}
    for n in names:
        for sl in slot_ids:
            x[(n, sl)] = model.NewBoolVar(f"sd_x_{n}_{sl}")

    y: Dict[str, cp_model.IntVar] = {}
    for n in names:
        var = model.NewBoolVar(f"sd_y_{n}")
        y[n] = var
        model.Add(sum(x[(n, sl)] for sl in slot_ids) == var)

    # Fill each slot with exactly one eligible player
    for sl, slot in zip(slot_ids, req.slots):
        for n in names:
            if by_name[n].pos not in set(slot.eligible):
                model.Add(x[(n, sl)] == 0)
        model.Add(sum(x[(n, sl)] for n in names) == 1)

    # Each player at most one slot
    for n in names:
        model.Add(sum(x[(n, sl)] for sl in slot_ids) <= 1)

    # Salary cap (MVP/CPT use slot-adjusted salary)
    model.Add(
        sum(
            x[(n, sl)] * _sd_salary_for_slot(by_name[n], id_to_label[sl])
            for n in names for sl in slot_ids
        ) <= req.cap
    )

    # -------- slot-aware locks/excludes (DK: CPT/FLEX, FD: MVP/FLEX) ----------
    excl_raw = set(req.excludes or [])
    lock_raw = set(req.locks or [])

    # split into name-wide and slot-specific ("Name::CPT" / "Name::FLEX" / "Name::MVP")
    name_excl = {k for k in excl_raw if "::" not in k}
    name_lock = {k for k in lock_raw if "::" not in k}

    slot_excl: Set[Tuple[str, str]] = set()
    slot_lock: Set[Tuple[str, str]] = set()
    for k in excl_raw:
        if "::" in k:
            nm, lab = k.split("::", 1)
            slot_excl.add((nm, lab.upper()))
    for k in lock_raw:
        if "::" in k:
            nm, lab = k.split("::", 1)
            slot_lock.add((nm, lab.upper()))

    # name-wide first
    for n in names:
        if n in name_excl:
            model.Add(y[n] == 0)
            for sl in slot_ids:
                model.Add(x[(n, sl)] == 0)
        if n in name_lock:
            model.Add(y[n] == 1)  # slot chosen by solver unless slot-locked below

    # canonical tag -> engine slot ids
    tag_to_slots: Dict[str, List[str]] = {}
    for sl in slot_ids:
        tag = id_to_label[sl].upper()  # FLEX1/2 -> FLEX; MVP/CPT passthrough
        tag_to_slots.setdefault(tag, []).append(sl)

    # slot-tagged excludes
    for n, tag in slot_excl:
        for sl in tag_to_slots.get(tag, []):
            model.Add(x[(n, sl)] == 0)

    # slot-tagged locks (single-slot: ==1; multi-slot like FLEX: exactly one & block others)
    for n, tag in slot_lock:
        slots = tag_to_slots.get(tag, [])
        if not slots:
            continue
        if len(slots) == 1:
            model.Add(x[(n, slots[0])] == 1)
        else:
            model.Add(sum(x[(n, sl)] for sl in slots) == 1)
            for sl in slot_ids:
                if sl not in slots:
                    model.Add(x[(n, sl)] == 0)

    # per-player exposure caps already hit across the run
    for n in names:
        if used_player_counts.get(n, 0) >= player_caps.get(n, 10**9):
            model.Add(y[n] == 0)

    # ===== per-tag exposure caps (MIN/MAX) using canonical label keys =====
    for n in names:
        for label, slots in tag_to_slots.items():
            key = f"{n}::{label}"
            cap_here = tag_caps.get(key, 10**9)
            used_here = used_tag_counts.get(key, 0)
            if used_here >= cap_here:
                for sl in slots:
                    model.Add(x[(n, sl)] == 0)

    # forced include for min% (supports "Name" and "Name::CPT/MVP/FLEX")
    if forced_include:
        if "::" in forced_include:
            nm, lab = forced_include.split("::", 1)
            lab = lab.upper()
            if (nm in names) and (lab in tag_to_slots):
                slots = tag_to_slots[lab]
                if len(slots) == 1:
                    model.Add(x[(nm, slots[0])] == 1)
                else:
                    model.Add(sum(x[(nm, sl)] for sl in slots) == 1)
                    for sl in slot_ids:
                        if sl not in slots:
                            model.Add(x[(nm, sl)] == 0)
        elif forced_include in y:
            model.Add(y[forced_include] == 1)

    # lineup uniqueness (Hamming distance)
    roster_size = len(slot_ids)
    for lineup in prior_lineups:
        model.Add(sum(y[n] for n in lineup if n in y) <= roster_size - max(1, req.min_diff))

    # lineup pOWN cap (percentage points)
    if isinstance(req.lineup_pown_max, (int, float)) and req.lineup_pown_max is not None:
        lhs = sum(
            int(round(
                (
                    by_name[n].cap_pown if (id_to_label[sl] in ("CPT","MVP") and by_name[n].cap_pown is not None)
                    else by_name[n].pown
                ) * 100
            )) * x[(n, sl)]
            for n in names for sl in slot_ids
        )
        model.Add(lhs <= int(round(req.lineup_pown_max)))

    # ---- NEW: require at least two distinct teams in the lineup ----
    all_teams = sorted({by_name[n].team for n in names if by_name[n].team})
    if all_teams:
        z_team: Dict[str, cp_model.IntVar] = {}
        for t in all_teams:
            z = model.NewBoolVar(f"sd_team_{t}")
            z_team[t] = z
            members = [n for n in names if by_name[n].team == t]
            if not members:
                model.Add(z == 0)
            else:
                # z_t == 1  <=> any player from team t is selected
                model.Add(sum(y[n] for n in members) >= z)
                model.Add(sum(y[n] for n in members) <= len(members) * z)
        # At least two teams represented
        model.Add(sum(z_team.values()) >= 2)

    # IF → THEN rules
    opp_map = _sd_team_to_opp_map(req.players)
    for ridx, r in enumerate(req.rules or []):
        # case-insensitive slot match so FD's MVP works
        if_slot_ids = [sl for sl in slot_ids if id_to_label[sl].upper() == str(r.if_slot).upper()]
        if not if_slot_ids:
            continue
        want_if_pos = set(r.if_pos)
        want_from_pos = set(r.from_pos)
        for if_sl in if_slot_ids:
            g = model.NewBoolVar(f"sd_rule_{ridx}_gate_{if_sl}")
            if_candidates = []
            for n in names:
                p = by_name[n]
                if p.pos in want_if_pos and (not r.if_team_exact or p.team == r.if_team_exact):
                    if_candidates.append(x[(n, if_sl)])
            if not if_candidates:
                model.Add(g == 0)
            else:
                s_if = sum(if_candidates)
                model.Add(s_if >= g)
                model.Add(s_if <= 1)
            if r.team_scope == "any":
                pool = [y[n] for n in names if by_name[n].pos in want_from_pos]
                if pool:
                    model.Add(sum(pool) >= r.then_at_least * g)
            elif r.team_scope == "exact_team" and r.team_exact:
                pool = [y[n] for n in names if (by_name[n].team == r.team_exact and by_name[n].pos in want_from_pos)]
                if pool:
                    model.Add(sum(pool) >= r.then_at_least * g)
            else:
                for n in names:
                    p = by_name[n]
                    if p.pos not in want_if_pos:
                        continue
                    if r.if_team_exact and p.team != r.if_team_exact:
                        continue
                    is_this = x[(n, if_sl)]
                    helper_vars: List[cp_model.IntVar] = []
                    if r.team_scope == "same_team":
                        for m in names:
                            q = by_name[m]
                            if q.team == p.team and q.pos in want_from_pos:
                                helper_vars.append(y[m])
                    elif r.team_scope == "opp_team":
                        opps = opp_map.get(p.team, set())
                        for m in names:
                            q = by_name[m]
                            if q.team in opps and q.pos in want_from_pos:
                                helper_vars.append(y[m])
                    if helper_vars:
                        model.Add(sum(helper_vars) >= r.then_at_least * is_this)

    # Objective
    model.Maximize(
        sum(
            int(_sd_metric_for_slot(by_name[n], id_to_label[sl], req.objective) * 1000) * x[(n, sl)]
            for n in names for sl in slot_ids
        )
    )

    # Solve
    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res != cp_model.OPTIMAL and res != cp_model.FEASIBLE:
        return None

    chosen = [n for n in names if solver.Value(y[n]) == 1]
    if len(chosen) != len(set(chosen)) or len(chosen) != len(slot_ids):
        return None

    slot_map: Dict[str, str] = {}
    for sl in slot_ids:
        for n in names:
            if solver.Value(x[(n, sl)]) == 1:
                slot_map[sl] = n
                break

    salary = sum(_sd_salary_for_slot(by_name[name], id_to_label[sl]) for sl, name in slot_map.items())
    total  = sum(_sd_metric_for_slot(by_name[name], id_to_label[sl], req.objective) for sl, name in slot_map.items())

    return chosen, salary, total, slot_map

# ---- Endpoints ----------------------------------------------------------

def _sd_scores(req: SDSolveRequest) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in req.players:
        out[p.name] = _sd_score(p, req.objective, (req.boosts or {}).get(p.name, 0), req.randomness)
    return out

@app.post("/solve_showdown_stream")
def solve_showdown_stream(req: SDSolveRequest):
    random.seed()
    used_player_counts: Dict[str, int] = {}
    used_tag_counts: Dict[str, int] = {}
    player_caps = _sd_player_caps(req)
    tag_caps    = _sd_player_caps_tag(req)
    prior: List[List[str]] = []

    def gen():
        produced = 0
        for i in range(req.n):
            scores = _sd_scores(req)
            need_tag = _sd_min_need_tag(req, used_tag_counts)
            forced = need_tag or _sd_min_need_player(req, used_player_counts)
            ans = _sd_solve_one(req, scores, used_player_counts, player_caps,
                                forced, prior, used_tag_counts, tag_caps)
            if not ans:
                yield sse_event({"done": True, "reason": "no_more_unique_or_exposure_capped", "produced": produced})
                return
            chosen, salary, total, slot_map = ans
            for n in chosen:
                used_player_counts[n] = used_player_counts.get(n, 0) + 1
            for sl, n in (slot_map or {}).items():
                label = _sd_canonical_label(sl)   # FLEX1 -> FLEX, MVP/CPT unchanged
                key = f"{n}::{label}"
                used_tag_counts[key] = used_tag_counts.get(key, 0) + 1
            prior.append(chosen)
            produced += 1
            ordered = _sd_ordered_players(slot_map)
            yield sse_event({
                "index": i + 1,
                "drivers": ordered,
                "salary": salary,
                "total": total,
            })
        yield sse_event({"done": True, "produced": produced})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@app.post("/solve_showdown")
def solve_showdown(req: SDSolveRequest):
    """
    Batch variant (returns all lineups in one JSON).
    """
    random.seed()

    used_player_counts: Dict[str, int] = {}
    used_tag_counts: Dict[str, int] = {}
    player_caps = _sd_player_caps(req)
    tag_caps    = _sd_player_caps_tag(req)

    prior: List[List[str]] = []
    out = []

    for _ in range(req.n):
        scores = _sd_scores(req)
        need_tag = _sd_min_need_tag(req, used_tag_counts)
        forced = need_tag or _sd_min_need_player(req, used_player_counts)

        ans = _sd_solve_one(req, scores, used_player_counts, player_caps,
                            forced, prior, used_tag_counts, tag_caps)
        if not ans:
            break

        chosen, salary, total, slot_map = ans

        for n in chosen:
            used_player_counts[n] = used_player_counts.get(n, 0) + 1
        for sl, n in (slot_map or {}).items():
            label = _sd_canonical_label(sl)   # FLEX1 -> FLEX
            key = f"{n}::{label}"
            used_tag_counts[key] = used_tag_counts.get(key, 0) + 1

        prior.append(chosen)
        ordered = _sd_ordered_players(slot_map)
        out.append({"drivers": ordered, "salary": salary, "total": total})

    return {"lineups": out, "produced": len(out)}
# ========================== END SHOWDOWN SOLVER ============================

# === MLB MODELS & SOLVER — DROP-IN (stack teams + pitchers kept) ===
# Relies on: cp_model, random, StreamingResponse, sse_event(...), SSE_HEADERS

from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field, root_validator

# ----------------------------- MLB models -----------------------------
class MLBPlayer(BaseModel):
    name: str
    team: str
    opp: str
    eligible: List[str]          # e.g., ["1B","OF"] or ["P"] (SP/RP are treated as "P")
    salary: int
    proj: float
    floor: float = 0.0
    ceil: float = 0.0
    pown: float = 0.0            # 0..1
    opt: float = 0.0             # 0..1

class MLBSlot(BaseModel):
    name: str
    eligible: List[str]          # allowed positions for this slot (e.g., ["2B"], ["C","1B"], ["UTIL"])

class SolveMLBRequest(BaseModel):
    site: Literal["dk","fd"]
    slots: List[MLBSlot]
    players: List[MLBPlayer]
    n: int
    cap: int
    objective: Literal["proj","floor","ceil","pown","opt"] = "proj"

    # shared controls
    locks: List[str] = Field(default_factory=list)
    excludes: List[str] = Field(default_factory=list)
    boosts: Dict[str, int] = Field(default_factory=dict)    # ± steps; 1 step = ±3%
    randomness: float = 0.0                                  # 0..100 (%)
    global_max_pct: float = 100.0                            # overall exposure cap
    min_pct: Dict[str, float] = Field(default_factory=dict)  # per-player min exposure
    max_pct: Dict[str, float] = Field(default_factory=dict)  # per-player max exposure
    min_diff: int = 1                                        # lineup uniqueness (Hamming)
    time_limit_ms: int = 1500

    # MLB-specific
    primary_stack_size: int = 5            # hitters only
    secondary_stack_size: int = 3          # hitters only, must differ from primary
    avoid_hitters_vs_opp_pitcher: bool = True
    max_hitters_vs_opp_pitcher: int = 0    # when avoid_* True, cap opposing hitters count

    # Cap on SUM of lineup pOWN percentage points (e.g., 200 means 200 total pOWN points)
    lineup_pown_max: Optional[float] = None

    # Accept legacy/frontend alias and arbitrary extras safely
    max_lineup_pown_pct: Optional[float] = None
    # Optional: distinct teams requirement (default: FD=3, DK=2 for hitters+pitchers total)
    min_distinct_teams: Optional[int] = None

    # 🔹 limit hitter pool & stack teams to this set (pitchers always allowed)
    allowed_teams: List[str] = Field(default_factory=list)

    class Config:
        extra = "ignore"  # ignore unknown fields from the client

    @root_validator(pre=True)
    def _unify_lineup_pown_cap(cls, values):
        if values.get("lineup_pown_max") is None and values.get("max_lineup_pown_pct") is not None:
            values["lineup_pown_max"] = values.get("max_lineup_pown_pct")
        return values

# ----------------------------- helpers -----------------------------
def _mlb_metric(p: MLBPlayer, objective: str) -> float:
    if objective == "proj": return p.proj
    if objective == "floor": return p.floor
    if objective == "ceil":  return p.ceil
    if objective == "pown":  return p.pown * 100.0
    if objective == "opt":   return p.opt  * 100.0
    return p.proj

def _mlb_score(p: MLBPlayer, objective: str, boost_steps: int, randomness_pct: float) -> float:
    base = _mlb_metric(p, objective)
    boosted = base * (1.0 + 0.03 * (boost_steps or 0))
    if randomness_pct > 0.0:
        r = randomness_pct / 100.0
        boosted *= (1.0 + random.uniform(-r, r))
    return boosted

def _cap_counts(req: SolveMLBRequest) -> Dict[str, int]:
    """Per-player max counts from global & per-player caps (can be zero)."""
    N = max(1, int(req.n))
    gcap = int(min(max(req.global_max_pct, 0.0), 100.0) / 100.0 * N)
    caps: Dict[str, int] = {}
    for p in req.players:
        per = int(min(max(req.max_pct.get(p.name, 100.0), 0.0), 100.0) / 100.0 * N)
        caps[p.name] = min(per, gcap) if gcap > 0 else 0
    return caps

def _mins_needed(req: SolveMLBRequest, counts: Dict[str, int]) -> Optional[str]:
    """Pick a player under min% exposure to force include (highest proj first)."""
    needs = []
    N = max(1, int(req.n))
    for p in req.players:
        need = int(min(max(req.min_pct.get(p.name, 0.0), 0.0), 100.0) / 100.0 * N)
        if counts.get(p.name, 0) < need:
            needs.append(p)
    if not needs:
        return None
    needs.sort(key=lambda r: r.proj, reverse=True)
    return needs[0].name

# ----------------------------- core solver -----------------------------
def _solve_one_mlb(
    req: SolveMLBRequest,
    scores: Dict[str, float],
    used: Dict[str, int],
    caps: Dict[str, int],
    forced_include: Optional[str],
    prior_lineups: List[List[str]],
):
    model = cp_model.CpModel()

    # Normalize positions: treat SP/RP as P
    def norm_elig(e: List[str]) -> set:
        out = set()
        for pos in (e or []):
            s = str(pos or "").upper().strip()
            if s in {"SP", "RP"}: s = "P"
            out.add(s)
        return out

    by_name: Dict[str, MLBPlayer] = {p.name: p for p in req.players}
    names = list(by_name.keys())
    slot_names = [s.name for s in req.slots]

    # decision vars: x[(name,slot)] and y[name]
    x: Dict[tuple, cp_model.IntVar] = {}
    for n in names:
        for sl in slot_names:
            x[(n, sl)] = model.NewBoolVar(f"x_{n}_{sl}")
    y: Dict[str, cp_model.IntVar] = {n: model.NewBoolVar(f"y_{n}") for n in names}
    for n in names:
        model.Add(sum(x[(n, sl)] for sl in slot_names) == y[n])

    # slot fill & eligibility
    for sl, slot in zip(slot_names, req.slots):
        slot_ok_pos = {p.upper() for p in (slot.eligible or [])}
        for n in names:
            elig = norm_elig(by_name[n].eligible)
            if not elig.intersection(slot_ok_pos):
                model.Add(x[(n, sl)] == 0)
        model.Add(sum(x[(n, sl)] for n in names) == 1)

    # at most one slot per player
    for n in names:
        model.Add(sum(x[(n, sl)] for sl in slot_names) <= 1)

    # salary cap
    model.Add(sum(y[n] * by_name[n].salary for n in names) <= req.cap)

    # -------- locks/excludes ----------
    excl = set(req.excludes or [])
    lock = set(req.locks or [])
    for n in names:
        if n in excl:
            model.Add(y[n] == 0)
            for sl in slot_names:
                model.Add(x[(n, sl)] == 0)
        if n in lock:
            model.Add(y[n] == 1)

    # exposure caps already hit across the run
    for n in names:
        if used.get(n, 0) >= caps.get(n, 10**9):
            model.Add(y[n] == 0)

    # forced include for min%
    if forced_include and forced_include in y:
        model.Add(y[forced_include] == 1)

    # lineup uniqueness (Hamming distance)
    roster_size = len(req.slots)
    for prev in prior_lineups:
        model.Add(sum(y[n] for n in prev if n in y) <= roster_size - max(1, req.min_diff))

    # Separate hitters and pitchers
    hitters = [n for n in names if "P" not in norm_elig(by_name[n].eligible)]
    pitchers = [n for n in names if "P" in norm_elig(by_name[n].eligible)]

    # 🔹 Restrict hitters to allowed teams (pitchers always allowed)
    allowed = {t.upper() for t in (req.allowed_teams or []) if t}
    if allowed:
        for n in hitters:
            if by_name[n].team.upper() not in allowed:
                model.Add(y[n] == 0)

    # Pitcher vs. opposing hitters restriction (optional)
    if req.avoid_hitters_vs_opp_pitcher:
        for p_name in pitchers:
            opp_team = by_name[p_name].opp
            if not opp_team:
                continue
            opp_hitters = [n for n in hitters if by_name[n].team == opp_team]
            if not opp_hitters:
                continue
            M = len(opp_hitters)
            s = sum(y[n] for n in opp_hitters)
            # If pitcher is chosen, limit chosen opposing hitters ≤ max_hitters_vs_opp_pitcher
            model.Add(s <= int(req.max_hitters_vs_opp_pitcher) + M * (1 - y[p_name]))

    # Build team set for hitter stacks (respect allowed_teams if provided)
    if allowed:
        teams = sorted({by_name[n].team for n in hitters if by_name[n].team and by_name[n].team.upper() in allowed})
    else:
        teams = sorted({by_name[n].team for n in hitters if by_name[n].team})

    # Pre-check which teams can meet stack sizes with current pool
    def can_meet(team: str, size: int) -> bool:
        if size <= 0: return True
        count = sum(1 for n in hitters if by_name[n].team == team)
        return count >= size

    # ---- Hitter-only stacking ----
    b_primary: Dict[str, cp_model.IntVar] = {}
    enforce_primary = req.primary_stack_size > 0 and any(can_meet(t, req.primary_stack_size) for t in teams)
    if enforce_primary:
        b_primary = {t: model.NewBoolVar(f"primary_{t}") for t in teams}
        for t, v in b_primary.items():
            t_hitters = [n for n in hitters if by_name[n].team == t]
            if t_hitters and can_meet(t, req.primary_stack_size):
                model.Add(sum(y[n] for n in t_hitters) >= req.primary_stack_size * v)
            else:
                model.Add(v == 0)
        # exactly one primary team only if feasible
        model.Add(sum(b_primary.values()) == 1)

    # Secondary stack: different team from primary (when feasible)
    enforce_secondary = req.secondary_stack_size > 0 and any(can_meet(t, req.secondary_stack_size) for t in teams)
    if enforce_secondary:
        b_secondary = {t: model.NewBoolVar(f"secondary_{t}") for t in teams}
        for t, v in b_secondary.items():
            t_hitters = [n for n in hitters if by_name[n].team == t]
            if t_hitters and can_meet(t, req.secondary_stack_size):
                model.Add(sum(y[n] for n in t_hitters) >= req.secondary_stack_size * v)
            else:
                model.Add(v == 0)
        if b_primary:
            for t in teams:
                # cannot select the same team for primary & secondary
                model.Add(b_primary[t] + b_secondary[t] <= 1)
            # only require one secondary if there exists a team distinct from primary that can meet it
            model.Add(sum(b_secondary.values()) == 1)
        else:
            model.Add(sum(b_secondary.values()) == 1)

    # Distinct teams (site rule; allow client override)
    min_distinct = req.min_distinct_teams
    if min_distinct is None:
        min_distinct = 3 if req.site == "fd" else 2
    if min_distinct and min_distinct > 1:
        all_teams = sorted({by_name[n].team for n in names if by_name[n].team})
        if all_teams:
            z_team = {t: model.NewBoolVar(f"team_sel_{t}") for t in all_teams}
            for t in all_teams:
                members = [n for n in names if by_name[n].team == t]
                if not members:
                    model.Add(z_team[t] == 0)
                    continue
                model.Add(sum(y[n] for n in members) >= z_team[t])
                model.Add(sum(y[n] for n in members) <= len(members) * z_team[t])
            model.Add(sum(z_team.values()) >= int(min_distinct))

    # Lineup pOWN% cap (percentage points, e.g., 200 means 200 total pOWN points)
    if isinstance(req.lineup_pown_max, (int, float)) and req.lineup_pown_max is not None:
        lhs = sum(int(round(by_name[n].pown * 100)) * y[n] for n in names)
        model.Add(lhs <= int(round(max(0.0, float(req.lineup_pown_max)))))

    # Objective
    model.Maximize(sum(int(scores[n] * 1000) * y[n] for n in names))

    # Solve
    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = [n for n in names if solver.Value(y[n]) == 1]
    if len(chosen) != len(req.slots):
        return None

    total = sum(_mlb_metric(by_name[n], req.objective) for n in chosen)
    salary = sum(by_name[n].salary for n in chosen)
    return chosen, salary, total

# ----------------------------- endpoints -----------------------------
@app.post("/solve_mlb_stream")
def solve_mlb_stream(req: SolveMLBRequest):
    random.seed()
    counts: Dict[str, int] = {}
    prior: List[List[str]] = []
    caps = _cap_counts(req)

    def gen():
        # small heartbeat so the client sees bytes immediately
        yield b":hb\n\n"
        produced = 0
        for i in range(req.n):
            # iteration scores with boosts + randomness
            scores: Dict[str, float] = {}
            for p in req.players:
                b = (req.boosts or {}).get(p.name, 0)
                scores[p.name] = _mlb_score(p, req.objective, b, req.randomness)

            include = _mins_needed(req, counts)

            ans = _solve_one_mlb(req, scores, counts, caps, include, prior)
            if not ans:
                yield sse_event({
                    "done": True,
                    "reason": "no_more_unique_or_exposure_capped",
                    "produced": produced,
                })
                return

            chosen, salary, total = ans
            for n in chosen:
                counts[n] = counts.get(n, 0) + 1
            prior.append(chosen)
            produced += 1

            yield sse_event({
                "index": i + 1,
                "drivers": sorted(chosen),
                "salary": salary,
                "total": total,
            })

        yield sse_event({"done": True, "produced": produced})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )

@app.post("/solve_mlb")
def solve_mlb(req: SolveMLBRequest):
    random.seed()
    counts: Dict[str, int] = {}
    prior: List[List[str]] = []
    caps = _cap_counts(req)
    out = []

    for _ in range(req.n):
        scores: Dict[str, float] = {}
        for p in req.players:
            b = (req.boosts or {}).get(p.name, 0)
            scores[p.name] = _mlb_score(p, req.objective, b, req.randomness)

        include = _mins_needed(req, counts)
        ans = _solve_one_mlb(req, scores, counts, caps, include, prior)
        if not ans:
            break

        chosen, salary, total = ans
        for n in chosen:
            counts[n] = counts.get(n, 0) + 1
        prior.append(chosen)
        out.append({"drivers": sorted(chosen), "salary": salary, "total": total})

    return {"lineups": out, "produced": len(out)}

# ======================= NASCAR (Cup/Xfinity/Trucks) =======================
# Matches the frontend:
# - POST /cup/solve_stream  (raw JSON chunks, one per lineup, separated by \n\n)
# - POST /cup/solve         (batch JSON)
# Payload keys: players[], roster, cap, n, objective, locks, excludes, boosts,
# randomness, global_max_pct, min_pct, max_pct, min_diff, time_limit_ms, groups[].
# Response keys: drivers[], salary, total

from typing import List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field
from ortools.sat.python import cp_model
from starlette.responses import StreamingResponse
import json, random

# ----- models (mirror the frontend) ----------------------------------
Site = Literal["dk","fd"]

class CupPlayer(BaseModel):
    driver: str
    salary: int
    proj: float
    floor: float = 0.0
    ceil: float  = 0.0
    pown: float  = 0.0   # 0..1
    opt: float   = 0.0   # 0..1
    class Config:
        extra = "ignore"

class CupGroup(BaseModel):
    mode: Literal["at_most","at_least","exactly"] = "at_most"
    count: int = 1
    players: List[str] = Field(default_factory=list)

class CupSolveRequest(BaseModel):
    site: Site = "dk"
    players: List[CupPlayer]
    roster: int
    cap: int
    n: int
    objective: Literal["proj","floor","ceil","pown","opt"] = "proj"

    locks: List[str] = Field(default_factory=list)
    excludes: List[str] = Field(default_factory=list)
    boosts: Dict[str, int] = Field(default_factory=dict)  # steps; 1 step = +3%
    randomness: float = 0.0                                # 0..100 (%)
    global_max_pct: float = 100.0
    min_pct: Dict[str, float] = Field(default_factory=dict)
    max_pct: Dict[str, float] = Field(default_factory=dict)
    min_diff: int = 1
    time_limit_ms: int = 1500

    groups: List[CupGroup] = Field(default_factory=list)

    # tolerated extra hint (the UI may add {series: "cup"})
    class Config:
        extra = "ignore"

# ----- helpers --------------------------------------------------------
def _cup_metric(p: CupPlayer, obj: str) -> float:
    if obj == "proj": return p.proj
    if obj == "floor": return p.floor
    if obj == "ceil":  return p.ceil
    if obj == "pown":  return p.pown * 100.0
    if obj == "opt":   return p.opt  * 100.0
    return p.proj

def _cup_score(p: CupPlayer, obj: str, rand_pct: float, boost_steps: int) -> float:
    base = _cup_metric(p, obj)
    boosted = base * (1.0 + 0.03 * (boost_steps or 0))
    if rand_pct > 0.0:
        r = rand_pct / 100.0
        boosted *= (1.0 + random.uniform(-r, r))
    return boosted

def _cup_cap_from_pct(N: int, pct: float) -> int:
    return int(max(0.0, min(100.0, pct)) / 100.0 * N)

def _cup_player_caps(req: CupSolveRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    gcap = _cup_cap_from_pct(N, req.global_max_pct)
    caps: Dict[str, int] = {}
    for p in req.players:
        per = _cup_cap_from_pct(N, req.max_pct.get(p.driver, 100.0))
        caps[p.driver] = min(per, gcap) if gcap > 0 else 0
    return caps

def _cup_min_need(req: CupSolveRequest, counts: Dict[str, int]) -> Optional[str]:
    N = max(1, int(req.n))
    needers = []
    for p in req.players:
        need = _cup_cap_from_pct(N, req.min_pct.get(p.driver, 0.0))
        if counts.get(p.driver, 0) < need:
            needers.append(p)
    if not needers:
        return None
    needers.sort(key=lambda r: r.proj, reverse=True)
    return needers[0].driver

def _cup_scores(req: CupSolveRequest) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in req.players:
        out[p.driver] = _cup_score(p, req.objective, req.randomness, (req.boosts or {}).get(p.driver, 0))
    return out

# ----- single-lineup solver ------------------------------------------
def _cup_solve_one(
    req: CupSolveRequest,
    scores: Dict[str, float],
    used_counts: Dict[str, int],
    caps: Dict[str, int],
    forced_include: Optional[str],
    prior: List[List[str]],
) -> Optional[Tuple[List[str], int, float]]:
    model = cp_model.CpModel()

    by_name = {p.driver: p for p in req.players}
    names = list(by_name.keys())
    roster = int(req.roster)

    # decision vars
    y: Dict[str, cp_model.IntVar] = {n: model.NewBoolVar(f"y_{n}") for n in names}

    # roster size & salary cap
    model.Add(sum(y[n] for n in names) == roster)
    model.Add(sum(y[n] * by_name[n].salary for n in names) <= req.cap)

    # locks/excludes
    excl = set(req.excludes or [])
    lock = set(req.locks or [])
    for n in names:
        if n in excl: model.Add(y[n] == 0)
        if n in lock: model.Add(y[n] == 1)

    # exposure caps already hit
    for n in names:
        if used_counts.get(n, 0) >= caps.get(n, 10**9):
            model.Add(y[n] == 0)

    # forced include for min%
    if forced_include and forced_include in y:
        model.Add(y[forced_include] == 1)

    # groups
    for g in req.groups or []:
        vars_in = [y[n] for n in g.players if n in y]
        if not vars_in: continue
        s = sum(vars_in)
        if g.mode == "at_most":   model.Add(s <= g.count)
        elif g.mode == "at_least":model.Add(s >= g.count)
        elif g.mode == "exactly": model.Add(s == g.count)

    # uniqueness vs prior (Hamming distance)
    for prev in prior:
        model.Add(sum(y[n] for n in prev if n in y) <= roster - max(1, req.min_diff))

    # objective
    model.Maximize(sum(int(scores[n] * 1000) * y[n] for n in names))

    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = [n for n in names if solver.Value(y[n]) == 1]
    if len(chosen) != roster:  # guard
        return None

    salary = sum(by_name[n].salary for n in chosen)
    total  = sum(_cup_metric(by_name[n], req.objective) for n in chosen)
    return chosen, salary, total

# ----- STREAM: raw JSON chunks separated by blank lines ----------------
@app.post("/cup/solve_stream")
def cup_solve_stream(req: CupSolveRequest):
    random.seed()

    counts: Dict[str, int] = {}
    caps = _cup_player_caps(req)
    prior: List[List[str]] = []

    def gen():
        produced = 0
        for _ in range(req.n):
            sc = _cup_scores(req)
            forced = _cup_min_need(req, counts)
            ans = _cup_solve_one(req, sc, counts, caps, forced, prior)
            if not ans:
                # send a 'done' chunk then stop
                yield (json.dumps({"done": True, "reason": "no_more_unique_or_exposure_capped", "produced": produced}) + "\n\n").encode()
                return
            chosen, salary, total = ans
            for n in chosen:
                counts[n] = counts.get(n, 0) + 1
            prior.append(chosen)
            produced += 1
            chunk = {"drivers": sorted(chosen), "salary": salary, "total": total}
            yield (json.dumps(chunk) + "\n\n").encode()
        yield (json.dumps({"done": True, "produced": produced}) + "\n\n").encode()

    # no SSE headers on purpose; client parses raw JSON parts
    return StreamingResponse(gen(), media_type="application/octet-stream")

# ----- BATCH -----------------------------------------------------------
@app.post("/cup/solve")
def cup_solve(req: CupSolveRequest):
    random.seed()

    counts: Dict[str, int] = {}
    caps = _cup_player_caps(req)
    prior: List[List[str]] = []
    out = []

    for _ in range(req.n):
        sc = _cup_scores(req)
        forced = _cup_min_need(req, counts)
        ans = _cup_solve_one(req, sc, counts, caps, forced, prior)
        if not ans:
            break
        chosen, salary, total = ans
        for n in chosen:
            counts[n] = counts.get(n, 0) + 1
        prior.append(chosen)
        out.append({"drivers": sorted(chosen), "salary": salary, "total": total})

    return {"lineups": out, "produced": len(out)}
# ======================= CUP CONTEST SIMULATOR (DROP-IN) =======================
# Endpoints:
#   POST /scale_payouts     -> {payouts:[[place, dollars], ...]}
#   POST /monte_carlo_sim   -> MonteResp (lineup + driver contest stats)

from typing import Any, List, Dict, Tuple, Optional, Literal
from fastapi import Body
from pydantic import BaseModel, Field
import random

# ---------- payout scaling ----------

class PayoutTemplate(BaseModel):
    paid_pct: float = 25.0  # % of field paid
    # [rank_fraction (0..1), x_buyin]; linear interpolate
    shape: List[List[float]] = Field(
        default_factory=lambda: [
            [0.00, 25.0],
            [0.01, 10.0],
            [0.05,  3.0],
            [0.10,  2.0],
            [0.25,  1.5],
        ]
    )

class ScalePayoutsReq(BaseModel):
    template: PayoutTemplate
    entries: int
    buy_in: float
    rake_pct: Optional[float] = None  # optional; not enforced tightly

class ScalePayoutsResp(BaseModel):
    payouts: List[List[float]]  # [[place, dollars], ...]

def _interp_xbuyin(ctrl: List[List[float]], frac: float) -> float:
    c = sorted(ctrl, key=lambda t: t[0])
    if not c:
        return 1.0
    if frac <= c[0][0]:
        return c[0][1]
    for i in range(1, len(c)):
        a, xa = c[i-1]
        b, xb = c[i]
        if frac <= b:
            t = 0.0 if b == a else (frac - a) / (b - a)
            return xa + t * (xb - xa)
    return c[-1][1]

@app.post("/scale_payouts", response_model=ScalePayoutsResp)
def scale_payouts(payload: ScalePayoutsReq = Body(...)):
    entries = max(1, int(payload.entries))
    buy_in = max(0.0, float(payload.buy_in))
    paid_pct = max(0.0, min(100.0, float(payload.template.paid_pct)))
    paid_places = max(1, int(round(entries * (paid_pct / 100.0))))

    out: List[List[float]] = []
    for place in range(1, paid_places + 1):
        frac = (place - 1) / max(1, entries - 1)  # 0 for 1st, ~1 for last
        xbuy = _interp_xbuyin(payload.template.shape, frac)
        amount = max(0.0, xbuy * buy_in)
        out.append([place, amount])
    return ScalePayoutsResp(payouts=out)

# ---------- contest simulator ----------

class SimPlayer(BaseModel):
    driver: str
    salary: int
    proj: float
    floor: float = 0.0
    ceil: float = 0.0
    pown: float = 0.0  # 0..1

class MonteReq(BaseModel):
    players: List[SimPlayer]
    entries: int
    buy_in: float
    payouts: List[List[float]]    # [[place, dollars], ...] from /scale_payouts
    sims: int = 1000
    randomness: float = 12.0      # % projection volatility
    site: Optional[Literal["dk","fd"]] = None
    roster: Optional[int] = None
    cap: Optional[int] = None
    # lineup pool build
    pool_size: int = 400
    build_time_limit_ms: int = 600

class LineupStat(BaseModel):
    lineup: List[str]
    avg_roi: float      # average ROI across all copies entered
    win_pct: float      # % of sims where any copy won
    top10_pct: float    # % of sims where any copy finished top 10%
    top1_pct: float     # % of sims where any copy finished top 1%
    cash_pct: float     # per-entry cash rate across all copies
    dupes_avg: float    # avg copies in the field WHEN this lineup appears

class DriverStat(BaseModel):
    driver: str
    avg_roi: float      # per-entry driver ROI (equal split)
    win_pct: float      # % of that driver's entries that finished 1st
    top10_pct: float    # % of entries in top 10%
    top1_pct: float     # % of entries in top 1%
    cash_pct: float     # % of entries that cashed

class MonteResp(BaseModel):
    lineups: List[LineupStat]
    drivers: List[DriverStat]
    sims: int

def _default_roster_and_cap(site: Optional[str]) -> Tuple[int, int]:
    if site == "fd":
        return 5, 50000
    return 6, 50000  # DK default

def _gauss_from_floor_ceil(mu: float, floor: float, ceil: float, jitter_pct: float) -> float:
    width = max(0.0, ceil - floor)
    sigma = max(0.1, width / 4.0)
    base = random.gauss(mu, sigma)
    if jitter_pct > 0:
        r = jitter_pct / 100.0
        base *= (1.0 + random.uniform(-r, r))
    return max(0.0, base)

def _build_lineup_pool(
    players: List[SimPlayer],
    roster: int,
    cap: int,
    pool_size: int,
    per_lineup_time_ms: int,
) -> List[List[str]]:
    """Use existing Cup CP-SAT to build a diverse pool by re-sampling scores."""
    cup_players = [CupPlayer(
        driver=p.driver, salary=int(p.salary), proj=float(p.proj),
        floor=float(p.floor), ceil=float(p.ceil), pown=float(p.pown), opt=0.0
    ) for p in players if p.salary > 0 and p.driver]

    counts: Dict[str, int] = {}
    caps = {p.driver: pool_size for p in cup_players}
    prior: List[List[str]] = []
    pool: List[List[str]] = []
    seen_keys: set = set()

    dummy_req = CupSolveRequest(
        site="dk", players=cup_players, roster=roster, cap=cap, n=1,
        objective="proj",
        locks=[], excludes=[], boosts={}, randomness=0.0,
        global_max_pct=100.0, min_pct={}, max_pct={}, min_diff=1,
        time_limit_ms=per_lineup_time_ms, groups=[]
    )

    for _ in range(pool_size * 3):  # extra attempts for uniqueness
        scores: Dict[str, float] = {}
        for p in cup_players:
            mu = p.proj
            width = max(0.0, p.ceil - p.floor)
            sigma = max(0.1, width / 4.0)
            s = random.gauss(mu, sigma) * (1.0 + random.uniform(-0.10, +0.10))
            scores[p.driver] = max(0.0, s)

        ans = _cup_solve_one(dummy_req, scores, counts, caps, forced_include=None, prior=prior)
        if not ans:
            break
        chosen, _, _ = ans
        key = "|".join(sorted(chosen))
        if key not in seen_keys:
            seen_keys.add(key)
            pool.append(chosen)
            prior.append(chosen)
            if len(pool) >= pool_size:
                break
    return pool

@app.post("/monte_carlo_sim", response_model=MonteResp)
def monte_carlo_sim(payload: MonteReq = Body(...)):
    random.seed()

    players = [p for p in payload.players if p.driver and p.salary > 0]
    if not players:
        return MonteResp(lineups=[], drivers=[], sims=0)

    roster, cap = (
        (int(payload.roster), int(payload.cap))
        if (payload.roster and payload.cap)
        else _default_roster_and_cap(payload.site)
    )

    # 1) build lineup pool
    pool = _build_lineup_pool(
        players=players,
        roster=roster,
        cap=cap,
        pool_size=max(50, int(payload.pool_size)),
        per_lineup_time_ms=max(100, int(payload.build_time_limit_ms)),
    )
    if not pool:
        return MonteResp(lineups=[], drivers=[], sims=0)

    # ownership weights
    eps = 1e-6
    by_name = {p.driver: p for p in players}
    def lineup_weight(L: List[str]) -> float:
        w = 1.0
        for d in L:
            w *= max(eps, float(by_name[d].pown) + eps)
        return w

    weights = [lineup_weight(L) for L in pool]
    totw = sum(weights) or 1.0
    probs = [w / totw for w in weights]

    entries = max(1, int(payload.entries))
    buy_in = max(0.0, float(payload.buy_in))
    sims = max(1, int(payload.sims))

    # payout lookup (exact places); ranks beyond last key -> $0
    payout_map = {int(place): float(amt) for place, amt in (payload.payouts or [])}
    max_paid_place = max(payout_map.keys(), default=0)

    top1_cut = max(1, int(round(entries * 0.01)))
    top10_cut = max(1, int(round(entries * 0.10)))

    n_pool = len(pool)
    plays_total = [0] * n_pool            # total copies across sims
    cash_plays  = [0] * n_pool            # copies that cashed
    win_sims    = [0] * n_pool            # sims where any copy won
    top1_sims   = [0] * n_pool            # sims where any copy was top 1%
    top10_sims  = [0] * n_pool            # sims where any copy was top 10%
    dup_sum     = [0] * n_pool            # total copies per sim (summed)
    appear_sims = [0] * n_pool            # sims where lineup appeared ≥1 time
    winnings    = [0.0] * n_pool          # total dollars won

    # driver accumulators (per entry)
    driver_keys = sorted({p.driver for p in players})
    d_idx = {d: i for i, d in enumerate(driver_keys)}
    d_plays = [0] * len(driver_keys)
    d_cash  = [0] * len(driver_keys)
    d_top1  = [0] * len(driver_keys)
    d_top10 = [0] * len(driver_keys)
    d_win   = [0] * len(driver_keys)
    d_win_sum = [0.0] * len(driver_keys)  # total $ won allocated equally per lineup member

    for _ in range(sims):
        # sample field with replacement
        sampled_idx: List[int] = random.choices(range(n_pool), weights=probs, k=entries)
        counts: Dict[int, int] = {}
        for pi in sampled_idx:
            counts[pi] = counts.get(pi, 0) + 1
        for pi, c in counts.items():
            dup_sum[pi] += c
            appear_sims[pi] += 1

        # draw one outcome for the race
        driver_pts = {d: _gauss_from_floor_ceil(by_name[d].proj, by_name[d].floor, by_name[d].ceil, payload.randomness)
                      for d in by_name.keys()}

        # score uniques once
        unique_scores: Dict[int, float] = {}
        for pi in counts.keys():
            s = sum(driver_pts[d] for d in pool[pi])
            unique_scores[pi] = s

        # expand to entries and rank
        ranked: List[Tuple[float, int]] = []
        for pi, c in counts.items():
            ranked.extend([(unique_scores[pi], pi)] * c)
        ranked.sort(key=lambda t: t[0], reverse=True)

        # best rank per lineup this sim
        best_rank: Dict[int, int] = {}

        # pay entries, accumulate stats
        for rank, (_, pi) in enumerate(ranked, start=1):
            plays_total[pi] += 1
            L = pool[pi]
            pay = payout_map.get(rank, 0.0) if rank <= max_paid_place else 0.0
            if pay > 0:
                cash_plays[pi] += 1

            winnings[pi] += pay
            if pi not in best_rank:
                best_rank[pi] = rank

            # driver per-entry stats & earnings split
            share = (pay / roster) if roster > 0 else 0.0
            for d in L:
                di = d_idx[d]
                d_plays[di] += 1
                d_win_sum[di] += share
                if pay > 0:
                    d_cash[di] += 1
                if rank == 1:
                    d_win[di] += 1
                if rank <= top1_cut:
                    d_top1[di] += 1
                if rank <= top10_cut:
                    d_top10[di] += 1

        # convert best-rank → lineup-level sim flags
        for pi, r in best_rank.items():
            if r == 1:
                win_sims[pi] += 1
            if r <= top1_cut:
                top1_sims[pi] += 1
            if r <= top10_cut:
                top10_sims[pi] += 1

    # build lineup stats
    lineup_stats: List[LineupStat] = []
    for pi, L in enumerate(pool):
        # avoid div-by-zero
        cost = plays_total[pi] * buy_in
        avg_roi = (winnings[pi] - cost) / cost if cost > 0 else 0.0
        cash_rate = (cash_plays[pi] / plays_total[pi]) * 100.0 if plays_total[pi] else 0.0
        dupes_avg = (dup_sum[pi] / appear_sims[pi]) if appear_sims[pi] else 0.0

        lineup_stats.append(LineupStat(
            lineup=L,
            avg_roi=avg_roi,
            win_pct=(win_sims[pi] / sims) * 100.0,
            top10_pct=(top10_sims[pi] / sims) * 100.0,
            top1_pct=(top1_sims[pi] / sims) * 100.0,
            cash_pct=cash_rate,
            dupes_avg=dupes_avg,
        ))

    lineup_stats.sort(key=lambda r: (r.avg_roi, r.top1_pct, r.cash_pct), reverse=True)

    # build driver stats (per-entry)
    driver_stats: List[DriverStat] = []
    cost_per_driver_entry = (buy_in / roster) if roster > 0 else buy_in
    for d, i in d_idx.items():
        if d_plays[i] == 0:
            driver_stats.append(DriverStat(driver=d, avg_roi=0.0, win_pct=0.0, top10_pct=0.0, top1_pct=0.0, cash_pct=0.0))
            continue
        earn = d_win_sum[i]
        spent = d_plays[i] * cost_per_driver_entry
        d_roi = (earn - spent) / spent if spent > 0 else 0.0
        driver_stats.append(DriverStat(
            driver=d,
            avg_roi=d_roi,
            win_pct=(d_win[i] / d_plays[i]) * 100.0,
            top10_pct=(d_top10[i] / d_plays[i]) * 100.0,
            top1_pct=(d_top1[i] / d_plays[i]) * 100.0,
            cash_pct=(d_cash[i] / d_plays[i]) * 100.0,
        ))

    driver_stats.sort(key=lambda r: (r.avg_roi, r.top1_pct, r.cash_pct), reverse=True)
    return MonteResp(lineups=lineup_stats, drivers=driver_stats, sims=sims)
# ===================== END CUP CONTEST SIMULATOR (DROP-IN) =====================
