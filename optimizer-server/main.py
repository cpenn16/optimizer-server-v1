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
) -> Optional[Tuple[List[str], int, float, str]]:
    """
    Returns (chosen_names, salary, total_metric, qb_team) or None.
    """
    model = cp_model.CpModel()

    # convenience maps
    by_name: Dict[str, NFLPlayer] = {p.name: p for p in req.players}
    names = list(by_name.keys())
    slot_names = [s.name for s in req.slots]

    # decision vars: x[name, slot] in {0,1}
    x: Dict[Tuple[str,str], cp_model.IntVar] = {}
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
    # offense positions:
    offense_pos = {"QB", "RB", "WR", "TE"}
    for n in names:
        p = by_name[n]
        if p.pos == "DST":
            # forbid offense against this DST (optionally only RB if flag)
            for m in names:
                q = by_name[m]
                if q.team == p.opp:
                    if req.avoid_offense_vs_opp_dst and q.pos in offense_pos:
                        model.Add(y[n] + y[m] <= 1)
                    elif req.avoid_rb_vs_opp_dst and q.pos == "RB":
                        model.Add(y[n] + y[m] <= 1)

    # stacking / bringback
    # exactly one QB slot exists → exactly one QB chosen
    qb_names = [n for n in names if by_name[n].pos == "QB"]
    if qb_names:
        model.Add(sum(y[n] for n in qb_names) == 1)

    # Build team -> list of player vars (by position), mapping for bringback
    team_to_players: Dict[str, Dict[str, List[str]]] = {}
    for n in names:
        p = by_name[n]
        team_to_players.setdefault(p.team, {}).setdefault(p.pos, []).append(n)

    team_to_opps = _team_to_opp_map(req.players)

    # Per-team rule resolver (overrides globals)
    def effective_rule(team: str) -> Tuple[int, int, bool, Optional[List[str]], Optional[int]]:
        r = _team_rule_for(team, req.team_stack_rules)
        qb_stack = r.qb_stack_min if r.qb_stack_min is not None else req.qb_stack_min
        bring_min = r.bringback_min if r.bringback_min is not None else req.bringback_min
        allow_rb = r.allow_rb_in_stack if r.allow_rb_in_stack is not None else req.stack_allow_rb
        bring_teams = r.bringback_teams
        max_team = r.max_from_team if r.max_from_team is not None else req.max_from_team
        return qb_stack, bring_min, allow_rb, bring_teams, max_team

    # indicate which team is the QB team
    qb_team_vars: Dict[str, cp_model.IntVar] = {}
    for team in team_to_players.keys():
        v = model.NewBoolVar(f"qb_team_{team}")
        qb_team_vars[team] = v
        # v == 1 if a QB from this team is chosen
        team_qbs = [n for n in qb_names if by_name[n].team == team]
        if team_qbs:
            model.Add(sum(y[n] for n in team_qbs) == v)
        else:
            model.Add(v == 0)

    # team exposure caps across the run (QB team)
    # If cap for team T is K lineups, and already used X, then forbid choosing QB from T when X>=K
    for team, cap in qb_team_caps.items():
        used = qb_team_caps_used.get(team, 0)
        if used >= cap:
            # force qb_team_vars[team] == 0
            if team in qb_team_vars:
                model.Add(qb_team_vars[team] == 0)

    # Apply stack & bringback with Big-M gating by qb_team_vars[team]
    for team, v in qb_team_vars.items():
        qb_stack, bring_min, allow_rb_in_stack, bring_teams, _max_team_ignored = effective_rule(team)

        # helpers on same team as QB
        helper_pos = {"WR", "TE"}
        if allow_rb_in_stack:
            helper_pos.add("RB")

        helpers_same = [n for n in names if by_name[n].team == team and by_name[n].pos in helper_pos]
        if helpers_same and qb_stack > 0:
            model.Add(sum(y[n] for n in helpers_same) >= qb_stack * v)

        # bringback from opponent(s)
        opp_teams: Set[str] = set()
        if bring_teams:
            opp_teams.update(bring_teams)
        else:
            # fallback to any seen opponents for this team on slate
            opp_teams.update(team_to_opps.get(team, set()))

        if bring_min > 0 and opp_teams:
            bring_pos = {"WR", "TE"}
            if allow_rb_in_stack:
                bring_pos.add("RB")
            br_vars = [y[n] for n in names if by_name[n].team in opp_teams and by_name[n].pos in bring_pos]
            if br_vars:
                model.Add(sum(br_vars) >= bring_min * v)

    # objective
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

    # compute salary / total / qb team
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
    """
    SSE endpoint expected by the React app.
    Emits: {index, drivers:[names], salary, total, done?}
    """
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

            # update exposures
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
    """
    Non-streaming fallback expected by the React app.
    """
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
