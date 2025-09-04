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

# ================== NFL SHOWDOWN MODELS & SOLVER ==================
# Drop-in replacement for your existing Showdown block.

from typing import List, Dict, Optional, Tuple, Set, Literal, Any
from pydantic import BaseModel, Field, validator
from ortools.sat.python import cp_model
import random

BasePos = Literal["QB", "RB", "WR", "TE", "DST", "K"]  # showdown supports K

class ShowdownSlot(BaseModel):
    name: Literal["CPT","MVP","FLEX"]
    eligible: List[BasePos]  # base positions only

class ShowdownPlayer(BaseModel):
    name: str
    pos: BasePos
    team: str
    opp: str

    # FLEX salary + metrics (always required)
    salary: int
    proj: float
    floor: float = 0.0
    ceil: float  = 0.0
    pown: float  = 0.0   # 0..1
    opt: float   = 0.0   # 0..1

    # Optional CPT/MVP overrides (if your feed provides them)
    cap_salary: Optional[int] = None
    cap_proj:    Optional[float] = None
    cap_floor:   Optional[float] = None
    cap_ceil:    Optional[float] = None
    cap_pown:    Optional[float] = None
    cap_opt:     Optional[float]  = None

    # Normalize common alt position spellings from feeds (helps FD/DST etc.)
    @validator("pos", pre=True)
    def _norm_pos(cls, v):
        s = str(v or "").upper().replace(" ", "")
        if s in {"DEF", "D", "D/ST", "DST"}:
            return "DST"
        if s in {"PK"}:
            return "K"
        return s

class IfThenRule(BaseModel):
    # Example: if CPT/MVP is QB then require at least 1 FLEX from {WR,TE}
    # scope: 'same_team' or 'opp_team' or 'any'
    if_tag: Literal["CPT","MVP"] = "CPT"
    if_pos: List[BasePos] = Field(default_factory=lambda: ["QB"])
    then_at_least: int = 1
    from_pos: List[BasePos] = Field(default_factory=lambda: ["WR","TE"])
    team_scope: Literal["any","same_team","opp_team"] = "any"

class SolveShowdownRequest(BaseModel):
    site: Literal["dk","fd"]
    # exactly 6 slots: 1 CPT/MVP + 5 FLEX
    slots: List[ShowdownSlot]
    players: List[ShowdownPlayer]
    n: int
    cap: int
    objective: Literal["proj","floor","ceil","pown","opt"] = "proj"

    # controls
    locks: List[str] = Field(default_factory=list)        # may include "::CPT"/"::FLEX"
    excludes: List[str] = Field(default_factory=list)     # may include "::CPT"/"::FLEX"
    boosts: Dict[str, int] = Field(default_factory=dict)  # +/- steps (3% per step)
    randomness: float = 0.0
    global_max_pct: float = 100.0
    min_pct: Dict[str, float] = Field(default_factory=dict)
    max_pct: Dict[str, float] = Field(default_factory=dict)
    min_diff: int = 0
    time_limit_ms: int = 1500

    # showdown specific
    max_overlap: int = 5  # FLEX-name overlap limit vs prior lineups
    lineup_pown_max: Optional[float] = None

    # IF→THEN rules
    rules: List[IfThenRule] = Field(default_factory=list)

# ---------- helpers ----------
def _sd_metric(p: ShowdownPlayer, tag: str, objective: str, mult: float) -> float:
    """
    Objective value for the player's tag (CPT/MVP vs FLEX).
    If tag is captain, prefer cap_* fields when present; else multiply base by mult (1.5).
    """
    if tag in ("CPT","MVP"):
        if objective == "proj" and p.cap_proj is not None:   return p.cap_proj
        if objective == "floor" and p.cap_floor is not None: return p.cap_floor
        if objective == "ceil" and p.cap_ceil is not None:   return p.cap_ceil
        if objective == "pown" and p.cap_pown is not None:   return p.cap_pown * 100.0
        if objective == "opt"  and p.cap_opt  is not None:   return p.cap_opt * 100.0
        base = p.proj if objective=="proj" else p.floor if objective=="floor" else p.ceil if objective=="ceil" else (p.pown*100.0 if objective=="pown" else p.opt*100.0)
        return base * mult
    # FLEX
    if objective == "proj":  return p.proj
    if objective == "floor": return p.floor
    if objective == "ceil":  return p.ceil
    if objective == "pown":  return p.pown * 100.0
    if objective == "opt":   return p.opt * 100.0
    return p.proj

def _sd_salary(p: ShowdownPlayer, tag: str, mult: float) -> int:
    if tag in ("CPT","MVP"):
        if p.cap_salary is not None:
            return int(p.cap_salary)
        return int(round(p.salary * mult))
    return int(p.salary)

def _caps_from_pct_sd(N: int, pct: float) -> int:
    return int(max(0.0, min(100.0, pct)) / 100.0 * N)

def _min_need_player_sd(req: SolveShowdownRequest, counts: Dict[str, int]) -> Optional[str]:
    needs = []
    N = max(1, int(req.n))
    for p in req.players:
        need = _caps_from_pct_sd(N, req.min_pct.get(p.name, 0.0))
        if counts.get(p.name, 0) < need:
            needs.append(p)
    if not needs:
        return None
    needs.sort(key=lambda r: r.proj, reverse=True)
    return needs[0].name

def _player_caps_sd(req: SolveShowdownRequest) -> Dict[str, int]:
    N = max(1, int(req.n))
    gcap = _caps_from_pct_sd(N, req.global_max_pct)
    caps = {}
    for p in req.players:
        per = _caps_from_pct_sd(N, req.max_pct.get(p.name, 100.0))
        caps[p.name] = min(per, gcap) if gcap > 0 else 0
    return caps

def _team_to_opp_map_sd(players: List[ShowdownPlayer]) -> Dict[str, Set[str]]:
    m: Dict[str, Set[str]] = {}
    for p in players:
        if p.team and p.opp:
            m.setdefault(p.team, set()).add(p.opp)
    return m

def _scores_for_iteration_sd(req: SolveShowdownRequest, cap_mult: float):
    # score[(name,tag)] for tag in {"CPT","FLEX"}
    scores: Dict[Tuple[str,str], float] = {}
    base_for_order: Dict[str, float] = {}
    for p in req.players:
        boost = (req.boosts or {}).get(p.name, 0)
        for tag in ("CPT", "FLEX"):
            base = _sd_metric(p, "CPT", req.objective, cap_mult) if tag == "CPT" else _sd_metric(p, "FLEX", req.objective, cap_mult)
            boosted = base * (1.0 + 0.03 * boost)
            if req.randomness > 0:
                r = req.randomness / 100.0
                boosted *= (1.0 + random.uniform(-r, r))
            scores[(p.name, tag)] = boosted
        base_for_order[p.name] = _sd_metric(p, "FLEX", req.objective, cap_mult)
    return scores, base_for_order

# ---------- core solver ----------
def _solve_one_showdown(
    req: SolveShowdownRequest,
    scores: Dict[Tuple[str,str], float],
    used_player_counts: Dict[str, int],
    player_caps: Dict[str, int],
    forced_include: Optional[str],
    prior_lineups: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    cap_mult = 1.5  # DK & FD both 1.5x
    model = cp_model.CpModel()

    by_name: Dict[str, ShowdownPlayer] = {p.name: p for p in req.players}
    names = list(by_name.keys())

    # Which base positions are allowed in CPT and FLEX from the provided slots
    cap_eligible: Set[BasePos] = set()
    flex_eligible: Set[BasePos] = set()
    found_cap = False
    for s in req.slots:
        if s.name in ("CPT","MVP") and not found_cap:
            cap_eligible.update(s.eligible)
            found_cap = True
        else:
            flex_eligible.update(s.eligible)
    if not cap_eligible: cap_eligible = {"QB","RB","WR","TE","DST","K"}
    if not flex_eligible: flex_eligible = {"QB","RB","WR","TE","DST","K"}

    cap  = {n: model.NewBoolVar(f"cap_{n}")  for n in names}
    flex = {n: model.NewBoolVar(f"flex_{n}") for n in names}
    chosen_any = {n: model.NewBoolVar(f"any_{n}") for n in names}

    for n in names:
        model.Add(chosen_any[n] == cap[n] + flex[n])
        model.Add(cap[n] + flex[n] <= 1)

    # roster
    model.Add(sum(cap[n] for n in names) == 1)
    model.Add(sum(flex[n] for n in names) == 5)

    # eligibility
    for n in names:
        if by_name[n].pos not in cap_eligible: model.Add(cap[n] == 0)
        if by_name[n].pos not in flex_eligible: model.Add(flex[n] == 0)

    # salary cap
    model.Add(
        sum(cap[n] * _sd_salary(by_name[n], "CPT", cap_mult) +
            flex[n] * _sd_salary(by_name[n], "FLEX", cap_mult) for n in names)
        <= req.cap
    )

    # locks/excludes (support "Name::CPT"/"Name::FLEX")
    excl_set = set(req.excludes or [])
    lock_set = set(req.locks or [])
    def want(name: str, tag: str) -> bool:
        key = f"{name}::{tag}"
        return key in lock_set or (name in lock_set and tag in ("CPT", "FLEX"))
    def ban(name: str, tag: str) -> bool:
        key = f"{name}::{tag}"
        return key in excl_set or (name in excl_set and tag in ("CPT", "FLEX"))

    for n in names:
        if want(n, "CPT"):  model.Add(cap[n] == 1)
        if want(n, "FLEX"): model.Add(flex[n] == 1)
        if ban(n, "CPT"):   model.Add(cap[n] == 0)
        if ban(n, "FLEX"):  model.Add(flex[n] == 0)

    # exposure caps across run
    for n in names:
        if used_player_counts.get(n, 0) >= player_caps.get(n, 10**9):
            model.Add(cap[n] == 0); model.Add(flex[n] == 0)

    # forced include to hit min%
    if forced_include and forced_include in by_name:
        model.Add(chosen_any[forced_include] == 1)

    # IF→THEN rules (gate on captain position/team)
    cap_pos_is: Dict[BasePos, cp_model.IntVar] = {}
    for pos in ("QB","RB","WR","TE","DST","K"):
        v = model.NewBoolVar(f"cap_is_{pos}")
        cap_pos_is[pos] = v
        model.Add(sum(cap[n] for n in names if by_name[n].pos == pos) == v)

    all_teams = sorted({by_name[n].team for n in names if by_name[n].team})
    cap_team_is: Dict[str, cp_model.IntVar] = {}
    for t in all_teams:
        v = model.NewBoolVar(f"cap_team_{t}")
        cap_team_is[t] = v
        model.Add(sum(cap[n] for n in names if by_name[n].team == t) == v)

    team_to_opp = _team_to_opp_map_sd(req.players)

    for rule in (req.rules or []):
        gate = model.NewBoolVar("rule_gate")
        model.Add(sum(cap_pos_is.get(pos, 0) for pos in rule.if_pos) >= 1).OnlyEnforceIf(gate)
        model.Add(sum(cap_pos_is.get(pos, 0) for pos in rule.if_pos) == 0).OnlyEnforceIf(gate.Not())

        def flex_vars_for_team_scope(scope: str, team: Optional[str] = None):
            if scope == "any" or not team:
                return [flex[n] for n in names if by_name[n].pos in rule.from_pos]
            if scope == "same_team":
                return [flex[n] for n in names if by_name[n].pos in rule.from_pos and by_name[n].team == team]
            if scope == "opp_team":
                opps = team_to_opp.get(team, set())
                return [flex[n] for n in names if by_name[n].pos in rule.from_pos and by_name[n].team in opps]
            return []

        if rule.team_scope == "any":
            vec = [flex[n] for n in names if by_name[n].pos in rule.from_pos]
            if vec:
                model.Add(sum(vec) >= rule.then_at_least).OnlyEnforceIf(gate)
        else:
            for t in all_teams:
                vec = flex_vars_for_team_scope(rule.team_scope, t)
                if not vec:
                    continue
                need = model.NewIntVar(0, 5, f"rule_need_{t}")
                model.Add(need == rule.then_at_least * cap_team_is[t])
                model.Add(sum(vec) >= need).OnlyEnforceIf(gate)

    # lineup pOWN cap (sum of tag-specific pOWN %, rounded to int)
    if isinstance(req.lineup_pown_max, (int, float)) and req.lineup_pown_max is not None:
        lhs = sum(
            int(round((by_name[n].cap_pown if by_name[n].cap_pown is not None else by_name[n].pown * 1.5) * 100)) * cap[n]
            + int(round(by_name[n].pown * 100)) * flex[n]
            for n in names
        )
        model.Add(lhs <= int(round(req.lineup_pown_max)))

    # ---- PREVENT EXACT DUPLICATES ----
    # Keep your existing "max_overlap" rule (FLEX-only), and also forbid a lineup
    # that matches both the previous CPT and all 5 FLEX exactly.
    for prev in prior_lineups:
        prev_cap = prev.get("cap", "")
        prev_flex = set(prev.get("flex", []))
        if prev_flex:
            # FLEX overlap limit
            model.Add(sum(flex[n] for n in names if n in prev_flex) <= max(0, int(req.max_overlap)))
        if prev_cap:
            # If CPT is same AND all five FLEX are the same -> would sum to 6; forbid that:
            model.Add(
                (cap[prev_cap] if prev_cap in cap else 0)
                + sum(flex[n] for n in names if n in prev_flex)
                <= 5
            )

    # objective
    model.Maximize(
        sum(int(scores[(n, "CPT")]  * 1000) * cap[n]  for n in names) +
        sum(int(scores[(n, "FLEX")] * 1000) * flex[n] for n in names)
    )

    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res != cp_model.OPTIMAL and res != cp_model.FEASIBLE:
        return None

    cap_name = next((n for n in names if solver.Value(cap[n]) == 1), "")
    flex_names = [n for n in names if solver.Value(flex[n]) == 1]
    if not cap_name or len(flex_names) != 5:
        return None

    total_salary = sum(
        _sd_salary(by_name[n], "CPT", cap_mult) if n == cap_name else _sd_salary(by_name[n], "FLEX", cap_mult)
        for n in [cap_name] + flex_names
    )
    total_metric = sum(
        _sd_metric(by_name[n], "CPT", req.objective, cap_mult) if n == cap_name else _sd_metric(by_name[n], "FLEX", req.objective, cap_mult)
        for n in [cap_name] + flex_names
    )
    return {"cap": cap_name, "flex": flex_names, "salary": total_salary, "total": total_metric}

# ---------- endpoints ----------
@app.post("/solve_showdown_stream")
def solve_showdown_stream(req: SolveShowdownRequest):
    random.seed()
    used_player_counts: Dict[str, int] = {}
    player_caps = _player_caps_sd(req)
    prior: List[Dict[str, Any]] = []

    def gen():
        produced = 0
        for i in range(req.n):
            scores, _ = _scores_for_iteration_sd(req, cap_mult=1.5)
            include = _min_need_player_sd(req, used_player_counts)
            ans = _solve_one_showdown(req, scores, used_player_counts, player_caps, include, prior)
            if not ans:
                yield sse_event({"done": True, "reason": "no_more_unique_or_exposure_capped", "produced": produced})
                return

            cap_name = ans["cap"]; flex_names = ans["flex"]
            # update exposures
            for n in [cap_name] + flex_names:
                used_player_counts[n] = used_player_counts.get(n, 0) + 1

            # remember for duplicate prevention & overlap rules
            prior.append({"cap": cap_name, "flex": flex_names})
            produced += 1

            yield sse_event({
                "index": i + 1,
                "drivers": [cap_name] + flex_names,  # first is CPT/MVP for UI
                "salary": ans["salary"],
                "total": ans["total"],
            })
        yield sse_event({"done": True, "produced": produced})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@app.post("/solve_showdown")
def solve_showdown(req: SolveShowdownRequest):
    random.seed()
    used_player_counts: Dict[str, int] = {}
    player_caps = _player_caps_sd(req)
    prior: List[Dict[str, Any]] = []
    out = []
    for _ in range(req.n):
        scores, _ = _scores_for_iteration_sd(req, cap_mult=1.5)
        include = _min_need_player_sd(req, used_player_counts)
        ans = _solve_one_showdown(req, scores, used_player_counts, player_caps, include, prior)
        if not ans:
            break
        cap_name = ans["cap"]; flex_names = ans["flex"]
        for n in [cap_name] + flex_names:
            used_player_counts[n] = used_player_counts.get(n, 0) + 1
        prior.append({"cap": cap_name, "flex": flex_names})
        out.append({"drivers": [cap_name] + flex_names, "salary": ans["salary"], "total": ans["total"]})
    return {"lineups": out, "produced": len(out)}

# === MLB MODELS & SOLVER — DROP-IN (no duplicate imports needed) ===

# NOTE: This block assumes you already have:
# - from starlette.responses import StreamingResponse
# - sse_event(payload: dict) -> bytes   # defined earlier in your file
# - cp_model imported
# - random imported

from typing import Optional, Literal
from pydantic import BaseModel, Field

# ----------------------------- MLB models -----------------------------
class MLBPlayer(BaseModel):
    name: str
    team: str
    opp: str
    eligible: list[str]  # e.g., ["1B","OF"] or ["P"] or ["SP","RP"] (we treat SP/RP as "P")
    salary: int
    proj: float
    floor: float = 0.0
    ceil: float = 0.0
    pown: float = 0.0  # 0..1
    opt: float = 0.0   # 0..1

class MLBSlot(BaseModel):
    name: str
    eligible: list[str]  # allowed positions for this slot (e.g., ["2B"], ["C","1B"], ["UTIL"...])

class SolveMLBRequest(BaseModel):
    site: Literal["dk","fd"]
    slots: list[MLBSlot]
    players: list[MLBPlayer]
    n: int
    cap: int
    objective: Literal["proj","floor","ceil","pown","opt"] = "proj"

    # controls (match NFL-style API)
    locks: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)
    boosts: dict[str, int] = Field(default_factory=dict)   # ± steps; 1 step = ±3%
    randomness: float = 0.0                                 # 0..100 (%)
    global_max_pct: float = 100.0                           # overall exposure cap
    min_pct: dict[str, float] = Field(default_factory=dict) # per-player min exposure
    max_pct: dict[str, float] = Field(default_factory=dict) # per-player max exposure
    min_diff: int = 1                                       # lineup uniqueness (Hamming)
    time_limit_ms: int = 1500

    # MLB-specific
    primary_stack_size: int = 5          # hitters only
    secondary_stack_size: int = 3        # hitters only, from a DIFFERENT team
    avoid_hitters_vs_opp_pitcher: bool = True
    max_hitters_vs_opp_pitcher: int = 0  # if avoid_* True, cap opposing hitters count
    lineup_pown_max: Optional[float] = None  # sum of pOWN% across lineup, in percentage points

# ----------------------------- helpers -----------------------------
def _mlb_metric(p: MLBPlayer, objective: str) -> float:
    if objective == "proj": return p.proj
    if objective == "floor": return p.floor
    if objective == "ceil": return p.ceil
    if objective == "pown": return p.pown * 100.0
    if objective == "opt":  return p.opt  * 100.0
    return p.proj

def _mlb_score(p: MLBPlayer, objective: str, boost_steps: int, randomness_pct: float) -> float:
    base = _mlb_metric(p, objective)
    boosted = base * (1.0 + 0.03 * (boost_steps or 0))
    if randomness_pct > 0.0:
        r = randomness_pct / 100.0
        boosted *= (1.0 + random.uniform(-r, r))
    return boosted

def _cap_counts(req: SolveMLBRequest) -> dict[str, int]:
    """Compute per-player max counts from global & per-player caps."""
    N = max(1, int(req.n))
    gcap = int(min(max(req.global_max_pct, 0.0), 100.0) / 100.0 * N)  # can be 0
    caps: dict[str, int] = {}
    for p in req.players:
        mp = req.max_pct.get(p.name, 100.0)
        per = int(min(max(mp, 0.0), 100.0) / 100.0 * N)
        caps[p.name] = min(per, gcap) if gcap > 0 else 0
    return caps

def _mins_needed(req: SolveMLBRequest, counts: dict[str, int]) -> Optional[str]:
    """Pick a player currently under min% exposure to force include."""
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
        for pos in e or []:
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
            allowed = bool(elig.intersection(slot_ok_pos))
            if not allowed:
                model.Add(x[(n, sl)] == 0)
        model.Add(sum(x[(n, sl)] for n in names) == 1)

    # at most one slot per player (already implied by y==sum(x), but keep hard)
    for n in names:
        model.Add(sum(x[(n, sl)] for sl in slot_names) <= 1)

    # salary cap
    model.Add(sum(y[n] * by_name[n].salary for n in names) <= req.cap)

    # locks/excludes
    excl = set(req.excludes or [])
    lock = set(req.locks or [])
    for n in names:
        if n in excl:
            model.Add(y[n] == 0)
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

    # Pitcher vs opposing hitters restriction (optional)
    if req.avoid_hitters_vs_opp_pitcher:
        for p_name in pitchers:
            opp_team = by_name[p_name].opp
            opp_hitters = [n for n in hitters if by_name[n].team == opp_team]
            if not opp_hitters:
                continue
            # If pitcher is chosen (y[p_name]==1), limit chosen opp hitters ≤ max_hitters_vs_opp_pitcher
            M = len(opp_hitters)
            s = sum(y[n] for n in opp_hitters)
            model.Add(s <= int(req.max_hitters_vs_opp_pitcher) + M * (1 - y[p_name]))

        # Hitter-only stacking — Primary (at least N hitters from exactly one team)
        teams = sorted({by_name[n].team for n in hitters})
        if req.primary_stack_size and req.primary_stack_size > 0 and teams:
            b_primary = {t: model.NewBoolVar(f"primary_{t}") for t in teams}
            for t, v in b_primary.items():
                t_hitters = [n for n in hitters if by_name[n].team == t]
                s = sum(y[n] for n in t_hitters)
                # If a team is the primary, it must have at least primary_stack_size hitters
                model.Add(s >= req.primary_stack_size * v)
            # Exactly one primary team
            model.Add(sum(b_primary.values()) == 1)
        else:
            b_primary = {}

        # Hitter-only stacking — Secondary (at least M hitters from exactly one team, different from primary)
        if req.secondary_stack_size and req.secondary_stack_size > 0 and teams:
            b_secondary = {t: model.NewBoolVar(f"secondary_{t}") for t in teams}
            for t, v in b_secondary.items():
                t_hitters = [n for n in hitters if by_name[n].team == t]
                s = sum(y[n] for n in t_hitters)
                # If a team is the secondary, it must have at least secondary_stack_size hitters
                model.Add(s >= req.secondary_stack_size * v)
            # Secondary must be different from the chosen primary
            if b_primary:
                for t in teams:
                    if t in b_primary:
                        model.Add(b_primary[t] + b_secondary[t] <= 1)
            # Exactly one secondary team
            model.Add(sum(b_secondary.values()) == 1)


    # FanDuel rule: lineup must include at least 3 different MLB teams (P can overlap with stacks)
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

    solver = cp_model.CpSolver()
    if req.time_limit_ms and req.time_limit_ms > 0:
        solver.parameters.max_time_in_seconds = req.time_limit_ms / 1000.0
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = [n for n in names if solver.Value(y[n]) == 1]
    total = sum(_mlb_metric(by_name[n], req.objective) for n in chosen)
    salary = sum(by_name[n].salary for n in chosen)
    # Sanity: exactly roster_size players
    if len(chosen) != len(req.slots):
        return None
    return chosen, salary, total

# ----------------------------- endpoints -----------------------------
@app.post("/solve_mlb_stream")
def solve_mlb_stream(req: SolveMLBRequest):
    random.seed()
    counts: dict[str, int] = {}
    prior: list[list[str]] = []
    caps = _cap_counts(req)

    def gen():
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
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
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
