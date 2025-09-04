# main.py
from typing import List, Dict, Optional, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from ortools.sat.python import cp_model
import random
import json


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod (e.g., ["https://your.site"])
    allow_credentials=False,      # "*" + credentials=True isn't spec-compliant
    allow_methods=["*"],
    allow_headers=["*"],
)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # critical for Nginx/Cloudflare
}


# ----------------------------- models -----------------------------
class Player(BaseModel):
    driver: str
    salary: int
    proj: float
    floor: float
    ceil: float
    pown: float  # 0..1
    opt: float   # 0..1


class GroupRule(BaseModel):
    mode: Literal["at_most", "at_least", "exactly"] = "at_most"
    count: int = 1
    players: List[str] = Field(default_factory=list)


class SolveRequest(BaseModel):
    players: List[Player]
    roster: int
    cap: int
    n: int
    objective: Literal["proj", "floor", "ceil", "pown", "opt"] = "proj"

    # constraints
    locks: List[str] = Field(default_factory=list)
    excludes: List[str] = Field(default_factory=list)
    boosts: Dict[str, int] = Field(default_factory=dict)   # +/- steps, 1 step = ±3%
    randomness: float = 0.0                                 # 0..100 (%)
    global_max_pct: float = 100.0                           # overall exposure cap (0..100)
    min_pct: Dict[str, float] = Field(default_factory=dict) # per-driver min exposure (0..100)
    max_pct: Dict[str, float] = Field(default_factory=dict) # per-driver max exposure (0..100)
    min_diff: int = 1                                       # Hamming distance between lineups
    time_limit_ms: int = 1500

    # optional extras
    groups: List[GroupRule] = Field(default_factory=list)   # player groups
    lineup_pown_max: Optional[float] = None                 # cap on SUM of lineup pOWN% (0..500-ish)


# --------------------------- helpers --------------------------
def metric_of(p: Player, objective: str) -> float:
    if objective == "proj":
        return p.proj
    if objective == "floor":
        return p.floor
    if objective == "ceil":
        return p.ceil
    if objective == "pown":
        return p.pown * 100.0  # report in %
    if objective == "opt":
        return p.opt * 100.0   # report in %
    return p.proj


def to_obj_score(p: Player, objective: str, boost_steps: int, randomness_pct: float) -> float:
    base = metric_of(p, objective)
    boosted = base * (1.0 + 0.03 * (boost_steps or 0))
    if randomness_pct > 0.0:
        r = randomness_pct / 100.0
        boosted *= (1.0 + random.uniform(-r, r))
    return boosted


def sse_event(payload: dict) -> bytes:
    # Server-Sent Events framing
    return f"event: progress\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


# ----------------------------- exposure math -----------------------------
def cap_counts(req: SolveRequest) -> Dict[str, int]:
    """
    Compute per-driver maximum counts (exposure caps) for the whole run.
    - Allows strict zeros (no 'or N' fallback).
    - global_max_pct is an upper bound on any per-driver cap.
    """
    N = max(1, int(req.n))
    gcap = int(min(max(req.global_max_pct, 0.0), 100.0) / 100.0 * N)  # can be 0
    caps: Dict[str, int] = {}
    for p in req.players:
        mp = req.max_pct.get(p.driver, 100.0)
        per = int(min(max(mp, 0.0), 100.0) / 100.0 * N)  # can be 0
        caps[p.driver] = min(per, gcap) if gcap > 0 else 0
    return caps


def mins_needed(req: SolveRequest, counts: Dict[str, int]) -> Optional[str]:
    """
    If any driver is below its min exposure target, pick the one with the
    highest raw projection and force-include them for the next lineup.
    """
    needs = []
    N = max(1, int(req.n))
    for p in req.players:
        need = int(min(max(req.min_pct.get(p.driver, 0.0), 0.0), 100.0) / 100.0 * N)
        if counts.get(p.driver, 0) < need:
            needs.append(p)
    if not needs:
        return None
    needs.sort(key=lambda r: r.proj, reverse=True)
    return needs[0].driver


# ----------------------------- solver -----------------------------
def solve_one_lineup(
    req: SolveRequest,
    scores: Dict[str, float],
    counts: Dict[str, int],
    cap_count: Dict[str, int],
    forced_include: Optional[str],
    prior_lineups: List[List[str]],
) -> Optional[List[str]]:

    model = cp_model.CpModel()
    x: Dict[str, cp_model.IntVar] = {}

    # Maps and names
    by_name = {p.driver: p for p in req.players}  # assumes unique driver names
    names = list(by_name.keys())

    # decision vars
    for name in names:
        x[name] = model.NewBoolVar(name)

    # roster & salary
    model.Add(sum(x[n] for n in names) == req.roster)
    model.Add(sum(x[n] * by_name[n].salary for n in names) <= req.cap)

    # excludes / locks
    for n in req.excludes:
        if n in x:
            model.Add(x[n] == 0)
    for n in req.locks:
        if n in x:
            model.Add(x[n] == 1)

    # exposure caps already reached
    for n in names:
        used = counts.get(n, 0)
        cap_for_n = cap_count.get(n, 10**9)
        if used >= cap_for_n:
            model.Add(x[n] == 0)

    # forced include (to satisfy min% over the run)
    if forced_include and forced_include in x:
        model.Add(x[forced_include] == 1)

    # min_diff vs prior lineups (Hamming distance)
    for lineup in prior_lineups:
        model.Add(sum(x[n] for n in lineup if n in x) <= req.roster - max(1, req.min_diff))

    # group rules
    for g in req.groups or []:
        group_vars = [x[n] for n in g.players if n in x]
        if not group_vars:
            continue
        s = sum(group_vars)
        if g.mode == "at_most":
            model.Add(s <= g.count)
        elif g.mode == "at_least":
            model.Add(s >= g.count)
        elif g.mode == "exactly":
            model.Add(s == g.count)

    # lineup pOWN% cap (Player.pown is 0..1, so convert to 0..100 percentage points)
    if isinstance(req.lineup_pown_max, (int, float)) and req.lineup_pown_max is not None:
        lhs = sum(int(round(by_name[n].pown * 100)) * x[n] for n in names)  # e.g., 0.25 -> 25
        model.Add(lhs <= int(round(req.lineup_pown_max)))  # cap is in "percentage points", e.g., 200 = 200%

    # objective
    model.Maximize(sum(int(scores[n] * 1000) * x[n] for n in names))

    # solve
    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res != cp_model.OPTIMAL and res != cp_model.FEASIBLE:
        return None

    chosen = [n for n in names if solver.Value(x[n]) == 1]
    # (BoolVars should prevent duplicates, but keep this sanity check)
    if len(chosen) != len(set(chosen)):
        return None
    return chosen


# ---------------------------- endpoints ----------------------------
@app.post("/solve_stream")
def solve_stream(req: SolveRequest):
    """
    Server-Sent Events: one lineup per 'progress' event.
    payload: { "index": i, "drivers": [...], "salary": 0, "total": 0.0 }
    """
    random.seed()

    counts: Dict[str, int] = {}
    prior: List[List[str]] = []
    caps = cap_counts(req)

    def gen():
        produced = 0
        for i in range(req.n):
            # per-iteration scores (apply boosts + randomness)
            scores: Dict[str, float] = {}
            for p in req.players:
                b = (req.boosts or {}).get(p.driver, 0)
                scores[p.driver] = to_obj_score(p, req.objective, b, req.randomness)

            include = mins_needed(req, counts)

            chosen = solve_one_lineup(req, scores, counts, caps, include, prior)
            if not chosen:
                yield sse_event({
                    "done": True,
                    "reason": "no_more_unique_or_exposure_capped",
                    "produced": produced,
                })
                return

            total = sum(
                metric_of(next(p for p in req.players if p.driver == n), req.objective)
                for n in chosen
            )
            salary = sum(next(p for p in req.players if p.driver == n).salary for n in chosen)

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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/solve")
def solve_batch(req: SolveRequest):
    """
    Non-streaming fallback: returns all lineups at once.
    """
    random.seed()

    counts: Dict[str, int] = {}
    prior: List[List[str]] = []
    caps = cap_counts(req)
    out = []

    for _ in range(req.n):
        scores: Dict[str, float] = {}
        for p in req.players:
            b = (req.boosts or {}).get(p.driver, 0)
            scores[p.driver] = to_obj_score(p, req.objective, b, req.randomness)

        include = mins_needed(req, counts)

        chosen = solve_one_lineup(req, scores, counts, caps, include, prior)
        if not chosen:
            break

        total = sum(
            metric_of(next(p for p in req.players if p.driver == n), req.objective)
            for n in chosen
        )
        salary = sum(next(p for p in req.players if p.driver == n).salary for n in chosen)

        for n in chosen:
            counts[n] = counts.get(n, 0) + 1
        prior.append(chosen)

        out.append({
            "drivers": sorted(chosen),
            "salary": salary,
            "total": total,
        })

    return {"lineups": out, "produced": len(out)}


# Optional: allow `python main.py` for quick testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)

# ================== NFL MODELS & SOLVER (slots/stacks) ==================
from typing import Tuple, Set

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

def _metric(p: NFLPlayer, objective: str) -> float:
    if objective == "proj": return p.proj
    if objective == "floor": return p.floor
    if objective == "ceil":  return p.ceil
    if objective == "pown":  return p.pown * 100.0
    if objective == "opt":   return p.opt * 100.0
    return p.proj

def _score(p: NFLPlayer, objective: str, boost_steps: int, randomness_pct: float) -> float:
    base = _metric(p, objective)
    boosted = base * (1.0 + 0.03 * (boost_steps or 0))
    if randomness_pct > 0:
        r = randomness_pct / 100.0
        boosted *= (1.0 + random.uniform(-r, r))
    return boosted

def _team_to_opp_map(players: List[NFLPlayer]) -> Dict[str, Set[str]]:
    """
    Build a mapping of team -> possible opponents on the slate.
    (There can be multiple if data includes multiple game times; that’s fine.)
    """
    m: Dict[str, Set[str]] = {}
    for p in players:
        if p.team and p.opp:
            m.setdefault(p.team, set()).add(p.opp)
    return m

def _caps_from_pct(N: int, pct: float) -> int:
    return int(max(0.0, min(100.0, pct)) / 100.0 * N)

def _team_rule_for(team: str, rules: List[TeamStackRule]) -> TeamStackRule:
    for r in rules or []:
        if r.team == team:
            return r
    return TeamStackRule(team=team)  # defaults (override later via globals)

def _solve_one_nfl(
    req: SolveNFLRequest,
    scores: Dict[str, float],
    used_player_counts: Dict[str, int],
    player_caps: Dict[str, int],
    forced_include: Optional[str],
    prior_lineups: List[List[str]],
    qb_team_caps_used: Dict[str, int],
    qb_team_caps: Dict[str, int],
):
    """
    Returns (chosen_names, salary, total_metric, qb_team) or None.
    """
    model = cp_model.CpModel()

    # convenience maps
    by_name: Dict[str, NFLPlayer] = {p.name: p for p in req.players}
    names = list(by_name.keys())
    slot_names = [s.name for s in req.slots]

    # decision vars: x[name, slot] in {0,1}
    x: Dict[tuple, cp_model.IntVar] = {}
    for n in names:
        for sl in slot_names:
            x[(n, sl)] = model.NewBoolVar(f"x_{n}_{sl}")

    # helper y[name] = chosen in any slot
    y: Dict[str, cp_model.IntVar] = {}
    for n in names:
        var = model.NewBoolVar(f"y_{n}")
        y[n] = var
        model.Add(sum(x[(n, sl)] for sl in slot_names) == var)

    # slot fill: exactly one per slot
    for sl, slot in zip(slot_names, req.slots):
        # only eligible can be placed into slot
        for n in names:
            if by_name[n].pos not in slot.eligible:
                model.Add(x[(n, sl)] == 0)
        model.Add(sum(x[(n, sl)] for n in names) == 1)

    # each player at most 1 slot (already ensured by y==sum x; but keep hardness)
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

    # per-player exposure caps already reached (across run)
    for n in names:
        if used_player_counts.get(n, 0) >= player_caps.get(n, 10**9):
            model.Add(y[n] == 0)

    # forced include (to hit min%)
    if forced_include and forced_include in y:
        model.Add(y[forced_include] == 1)

    # min_diff vs prior lineups
    roster_size = len(req.slots)
    for lineup in prior_lineups:
        model.Add(sum(y[n] for n in lineup if n in y) <= roster_size - max(1, req.min_diff))

    # group rules (player names)
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

    # team constraints
    # max_from_team (global)
    if isinstance(req.max_from_team, int) and req.max_from_team is not None:
        by_team: Dict[str, List[str]] = {}
        for n in names:
            by_team.setdefault(by_name[n].team, []).append(n)
        for t, members in by_team.items():
            model.Add(sum(y[n] for n in members) <= req.max_from_team)

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

    # stacking / bringback
    qb_names = [n for n in names if by_name[n].pos == "QB"]
    if qb_names:
        model.Add(sum(y[n] for n in qb_names) == 1)

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
        max_team = r.max_from_team if r.max_from_team is not None else req.max_from_team
        return qb_stack, bring_min, allow_rb, bring_teams, max_team

    qb_team_vars: Dict[str, cp_model.IntVar] = {}
    for team in team_to_players.keys():
        v = model.NewBoolVar(f"qb_team_{team}")
        qb_team_vars[team] = v
        team_qbs = [n for n in qb_names if by_name[n].team == team]
        if team_qbs:
            model.Add(sum(y[n] for n in team_qbs) == v)
        else:
            model.Add(v == 0)

    for team, v in qb_team_vars.items():
        pass  # team caps handled below in showdown solver only

    for team, v in qb_team_vars.items():
        qb_stack, bring_min, allow_rb_in_stack, bring_teams, _max_team_ignored = effective_rule(team)
        helper_pos = {"WR", "TE"}
        if allow_rb_in_stack:
            helper_pos.add("RB")

        helpers_same = [n for n in names if by_name[n].team == team and by_name[n].pos in helper_pos]
        if helpers_same and qb_stack > 0:
          model.Add(sum(y[n] for n in helpers_same) >= qb_stack * v)

        opp_teams = set(bring_teams or []) or set(team_to_opps.get(team, set()))
        if bring_min > 0 and opp_teams:
          bring_pos = {"WR", "TE"}
          if allow_rb_in_stack:
            bring_pos.add("RB")
          br_vars = [y[n] for n in names if by_name[n].team in opp_teams and by_name[n].pos in bring_pos]
          if br_vars:
            model.Add(sum(br_vars) >= bring_min * v)

    model.Maximize(sum(int(scores[n] * 1000) * y[n] for n in names))

    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res != cp_model.OPTIMAL and res != cp_model.FEASIBLE:
        return None

    chosen = [n for n in names if solver.Value(y[n]) == 1]
    if len(chosen) != len(set(chosen)) or len(chosen) != len(req.slots):
        return None

    salary = sum(by_name[n].salary for n in chosen)
    total = sum(_metric(by_name[n], req.objective) for n in chosen)
    qb_team = ""
    for team, v in qb_team_vars.items():
        if solver.Value(v) == 1:
            qb_team = team
            break

    return chosen, salary, total, qb_team

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

def _player_caps(req: SolveNFLRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    gcap = _caps_from_pct(N, req.global_max_pct)
    caps = {}
    for p in req.players:
        per = _caps_from_pct(N, req.max_pct.get(p.name, 100.0))
        caps[p.name] = min(per, gcap) if gcap > 0 else 0
    return caps

def _qb_team_caps(req: SolveNFLRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    out = {}
    for team, pct in (req.team_max_pct or {}).items():
        out[team] = _caps_from_pct(N, pct)
    return out

def _scores_for_iteration(req: SolveNFLRequest) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for p in req.players:
        b = (req.boosts or {}).get(p.name, 0)
        scores[p.name] = _score(p, req.objective, b, req.randomness)
    return scores

@app.post("/solve_nfl_stream")
def solve_nfl_stream(req: SolveNFLRequest):
    random.seed()

    used_player_counts: Dict[str, int] = {}
    prior: List[List[str]] = []
    player_caps = _player_caps(req)

    qb_team_caps_used: Dict[str, int] = {}
    qb_team_caps = _qb_team_caps(req)

    def gen():
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
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@app.post("/solve_nfl")
def solve_nfl(req: SolveNFLRequest):
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

    for sl, slot in zip(slot_ids, req.slots):
        for n in names:
            if by_name[n].pos not in set(slot.eligible):
                model.Add(x[(n, sl)] == 0)
        model.Add(sum(x[(n, sl)] for n in names) == 1)

    for n in names:
        model.Add(sum(x[(n, sl)] for sl in slot_ids) <= 1)

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

    # index tag -> engine slot ids
    tag_to_slots: Dict[str, List[str]] = {}
    for sl in slot_ids:
        tag = id_to_label[sl].upper()  # FLEX1/2 -> FLEX; MVP/CPT passthrough
        tag_to_slots.setdefault(tag, []).append(sl)

    # slot-tagged excludes
    for n, tag in slot_excl:
        for sl in tag_to_slots.get(tag, []):
            model.Add(x[(n, sl)] == 0)

    # slot-tagged locks (single-slot: ==1; multi-slot like FLEX: exactly one)
    for n, tag in slot_lock:
        slots = tag_to_slots.get(tag, [])
        if not slots:
            continue
        if len(slots) == 1:
            model.Add(x[(n, slots[0])] == 1)
        else:
            model.Add(sum(x[(n, sl)] for sl in slots) == 1)  # exactly one FLEX*
            for sl in slot_ids:                               # and forbid non-matching tags
                if sl not in slots:
                    model.Add(x[(n, sl)] == 0)

    # exposure caps already hit across the run
    for n in names:
        if used_player_counts.get(n, 0) >= player_caps.get(n, 10**9):
            model.Add(y[n] == 0)

    # per-tag caps (name::slotId)
    for n in names:
        for sl in slot_ids:
            key = f"{n}::{sl}"
            if used_tag_counts.get(key, 0) >= tag_caps.get(key, 10**9):
                model.Add(x[(n, sl)] == 0)

    # forced include for min%
    if forced_include:
        if "::" in forced_include:
            nm, sl = forced_include.split("::", 1)
            if (nm in names) and (sl in slot_ids):
                model.Add(x[(nm, sl)] == 1)
        elif forced_include in y:
            model.Add(y[forced_include] == 1)

    # lineup uniqueness (Hamming distance)
    roster_size = len(slot_ids)
    for lineup in prior_lineups:
        model.Add(sum(y[n] for n in lineup if n in y) <= roster_size - max(1, req.min_diff))

    # lineup pOWN max (integerized)
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

    model.Maximize(
        sum(
            int(_sd_metric_for_slot(by_name[n], id_to_label[sl], req.objective) * 1000) * x[(n, sl)]
            for n in names for sl in slot_ids
        )
    )

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
                key = f"{n}::{sl}"
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
            key = f"{n}::{sl}"
            used_tag_counts[key] = used_tag_counts.get(key, 0) + 1

        prior.append(chosen)
        ordered = _sd_ordered_players(slot_map)
        out.append({"drivers": ordered, "salary": salary, "total": total})

    return {"lineups": out, "produced": len(out)}
# ========================== END SHOWDOWN SOLVER ==========================


# === MLB MODELS & SOLVER — DROP-IN (stack teams + pitchers kept) ===
# Relies on: cp_model, random, StreamingResponse, sse_event(...), SSE_HEADERS

from typing import Optional, Literal
from pydantic import BaseModel, Field

# ----------------------------- MLB models -----------------------------
class MLBPlayer(BaseModel):
    name: str
    team: str
    opp: str
    eligible: list[str]          # e.g., ["1B","OF"] or ["P"] (SP/RP are treated as "P")
    salary: int
    proj: float
    floor: float = 0.0
    ceil: float = 0.0
    pown: float = 0.0            # 0..1
    opt: float = 0.0             # 0..1

class MLBSlot(BaseModel):
    name: str
    eligible: list[str]          # allowed positions for this slot (e.g., ["2B"], ["C","1B"], ["UTIL"])

class SolveMLBRequest(BaseModel):
    site: Literal["dk","fd"]
    slots: list[MLBSlot]
    players: list[MLBPlayer]
    n: int
    cap: int
    objective: Literal["proj","floor","ceil","pown","opt"] = "proj"

    # shared controls
    locks: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    boosts: dict[str, int] = Field(default_factory=dict)    # ± steps; 1 step = ±3%
    randomness: float = 0.0                                  # 0..100 (%)
    global_max_pct: float = 100.0                            # overall exposure cap
    min_pct: dict[str, float] = Field(default_factory=dict)  # per-player min exposure
    max_pct: dict[str, float] = Field(default_factory=dict)  # per-player max exposure
    min_diff: int = 1                                        # lineup uniqueness (Hamming)
    time_limit_ms: int = 1500

    # MLB-specific
    primary_stack_size: int = 5            # hitters only
    secondary_stack_size: int = 3          # hitters only, must differ from primary
    avoid_hitters_vs_opp_pitcher: bool = True
    max_hitters_vs_opp_pitcher: int = 0    # when avoid_* True, cap opposing hitters count
    lineup_pown_max: Optional[float] = None  # sum of pOWN% across lineup, in percentage points

    # 🔹 limit hitter pool & stack teams to this set (pitchers always allowed)
    allowed_teams: list[str] = Field(default_factory=list)

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

def _cap_counts(req: SolveMLBRequest) -> dict[str, int]:
    """Per-player max counts from global & per-player caps (can be zero)."""
    N = max(1, int(req.n))
    gcap = int(min(max(req.global_max_pct, 0.0), 100.0) / 100.0 * N)
    caps: dict[str, int] = {}
    for p in req.players:
        per = int(min(max(req.max_pct.get(p.name, 100.0), 0.0), 100.0) / 100.0 * N)
        caps[p.name] = min(per, gcap) if gcap > 0 else 0
    return caps

def _mins_needed(req: SolveMLBRequest, counts: dict[str, int]) -> Optional[str]:
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
    scores: dict[str, float],
    used: dict[str, int],
    caps: dict[str, int],
    forced_include: Optional[str],
    prior_lineups: list[list[str]],
):
    model = cp_model.CpModel()

    # Normalize positions: treat SP/RP as P
    def norm_elig(e: list[str]) -> set[str]:
        out = set()
        for pos in (e or []):
            s = str(pos or "").upper().strip()
            if s in {"SP", "RP"}: s = "P"
            out.add(s)
        return out

    by_name: dict[str, MLBPlayer] = {p.name: p for p in req.players}
    names = list(by_name.keys())
    slot_names = [s.name for s in req.slots]

    # decision vars: x[(name,slot)] and y[name]
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    for n in names:
        for sl in slot_names:
            x[(n, sl)] = model.NewBoolVar(f"x_{n}_{sl}")
    y: dict[str, cp_model.IntVar] = {n: model.NewBoolVar(f"y_{n}") for n in names}
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

    # at most one slot per player (redundant, keeps hardness)
    for n in names:
        model.Add(sum(x[(n, sl)] for sl in slot_names) <= 1)

    # salary cap
    model.Add(sum(y[n] * by_name[n].salary for n in names) <= req.cap)

    # -------- locks/excludes (name-wide) ----------
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

    # ---- Hitter-only stacking ----
    # Primary: at least primary_stack_size hitters from exactly one team
    b_primary = {}
    if req.primary_stack_size and req.primary_stack_size > 0 and teams:
        b_primary = {t: model.NewBoolVar(f"primary_{t}") for t in teams}
        for t, v in b_primary.items():
            t_hitters = [n for n in hitters if by_name[n].team == t]
            if t_hitters:
                model.Add(sum(y[n] for n in t_hitters) >= req.primary_stack_size * v)
            else:
                model.Add(v == 0)
        model.Add(sum(b_primary.values()) == 1)

    # Secondary: at least secondary_stack_size hitters from exactly one team, different from primary
    if req.secondary_stack_size and req.secondary_stack_size > 0 and len(teams) >= (2 if b_primary else 1):
        b_secondary = {t: model.NewBoolVar(f"secondary_{t}") for t in teams}
        for t, v in b_secondary.items():
            t_hitters = [n for n in hitters if by_name[n].team == t]
            if t_hitters:
                model.Add(sum(y[n] for n in t_hitters) >= req.secondary_stack_size * v)
            else:
                model.Add(v == 0)
        if b_primary:
            for t in teams:
                model.Add(b_primary[t] + b_secondary[t] <= 1)
        model.Add(sum(b_secondary.values()) == 1)

    # FanDuel: lineup must include at least 3 different MLB teams (P can overlap)
    if req.site == "fd":
        all_teams = sorted({by_name[n].team for n in names if by_name[n].team})
        if all_teams:
            z_team = {t: model.NewBoolVar(f"team_sel_{t}") for t in all_teams}
            for t in all_teams:
                members = [n for n in names if by_name[n].team == t]
                if not members:
                    model.Add(z_team[t] == 0)
                    continue
                # z_t == 1  <=>  any player from team t is selected
                model.Add(sum(y[n] for n in members) >= z_team[t])
                model.Add(sum(y[n] for n in members) <= len(members) * z_team[t])
            model.Add(sum(z_team.values()) >= 3)

    # Lineup pOWN% cap (percentage points, e.g., 200 means 200 total pOWN points)
    if isinstance(req.lineup_pown_max, (int, float)) and req.lineup_pown_max is not None:
        lhs = sum(int(round(by_name[n].pown * 100)) * y[n] for n in names)
        model.Add(lhs <= int(round(req.lineup_pown_max)))

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
    counts: dict[str, int] = {}
    prior: list[list[str]] = []
    caps = _cap_counts(req)

    def gen():
        # small heartbeat so the client sees bytes immediately
        yield b":hb\n\n"
        produced = 0
        for i in range(req.n):
            # iteration scores with boosts + randomness
            scores: dict[str, float] = {}
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
    counts: dict[str, int] = {}
    prior: list[list[str]] = []
    caps = _cap_counts(req)
    out = []

    for _ in range(req.n):
        scores: dict[str, float] = {}
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
