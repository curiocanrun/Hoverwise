"""
3D Path Planning Algorithms for Aerial Robotics
===============================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

Two geometric planners for a point robot whose obstacles are inflated by a safety
margin (the margin accounts for the quadcopter's physical radius plus an allowance for
trajectory deviation and tracking error):

  1. AStarPlanner3D - A* on a uniform voxel lattice with 26-connectivity and a Euclidean
     heuristic, followed by greedy line-of-sight shortcutting.  The lattice path is
     optimal *on the lattice graph*; it is not optimal in continuous space.
  2. RRTStarPlanner3D - RRT* (Karaman & Frazzoli, 2011) with goal-biased uniform
     sampling, choose-parent, rewiring with cost propagation to the rewired subtree, and
     the standard shrinking neighbourhood radius
         r_n = min( gamma * (ln n / n)^(1/3), eta ),   gamma > gamma*.
     Asymptotic optimality is a statement about n -> infinity; a run with a fixed
     iteration budget returns a feasible, improved but not optimal path.

Collision checking uses exact signed-distance functions (SDFs) of the obstacles
(sphere, finite vertical cylinder, axis-aligned box).  "Clearance" means the Euclidean
distance from a point to the nearest obstacle surface.  A point is valid when it lies in
the workspace bounds and its clearance is strictly greater than the safety margin; a
segment is valid when every point on it is valid (checked by Lipschitz bisection, see
Environment3D.is_segment_valid).

Both planners return (path, info).  `info["success"]` is True only for a complete,
collision-checked start-to-goal path.  On failure the returned path is EMPTY (shape
(0, 3)) and `info["status"]` says why: "invalid_start", "invalid_goal", "no_path"
(A* search exhausted or expansion limit) or "partial" (RRT* tree never reached the
goal; the collision-free branch towards the goal is in info["partial_path"]).
"""

import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from math import gamma as _gamma_fn

import numpy as np

DEFAULT_SAFETY_MARGIN = 0.5      # m, used when a planner is created without a margin
SEGMENT_CHECK_TOL = 1e-3         # m, bisection stops below this sub-segment length

STATUS_SUCCESS = "success"
STATUS_INVALID_START = "invalid_start"
STATUS_INVALID_GOAL = "invalid_goal"
STATUS_NO_PATH = "no_path"
STATUS_PARTIAL = "partial"

_EMPTY_PATH = np.zeros((0, 3), dtype=np.float64)


@dataclass
class Obstacle3D:
    """3D obstacle.

    kind 'sphere'  : centre `center`, radius `radius`.
    kind 'cylinder': vertical (z-axis) cylinder, radius `radius`, occupying
                     z in [center[2], center[2] + height]  (center[2] is the BASE).
    kind 'box'     : axis-aligned box centred at `center` with edge lengths `size`.
    """
    kind: str
    center: np.ndarray
    radius: float = 1.0
    height: float = 5.0
    size: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 5.0]))

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)
        if self.kind not in ("sphere", "cylinder", "box"):
            raise ValueError(f"unknown obstacle kind {self.kind!r}")

    def signed_distance(self, points) -> np.ndarray:
        """Exact signed distance from point(s) to the obstacle surface (negative inside).

        points: (3,) or (N, 3).  Returns a float or an (N,) array.
        """
        P = np.asarray(points, dtype=np.float64)
        single = P.ndim == 1
        P = np.atleast_2d(P)
        c = self.center
        if self.kind == "sphere":
            d = np.linalg.norm(P - c, axis=1) - self.radius
        elif self.kind == "cylinder":
            half_h = 0.5 * self.height
            zc = c[2] + half_h
            dr = np.linalg.norm(P[:, 0:2] - c[0:2], axis=1) - self.radius
            dz = np.abs(P[:, 2] - zc) - half_h
            outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
            inside = np.minimum(np.maximum(dr, dz), 0.0)
            d = outside + inside
        else:  # box
            q = np.abs(P - c) - 0.5 * self.size
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            d = outside + inside
        return float(d[0]) if single else d

    def is_in_collision(self, point: np.ndarray, safety_margin: float = DEFAULT_SAFETY_MARGIN) -> bool:
        """True if the point is closer than `safety_margin` to the obstacle (or inside it)."""
        return bool(self.signed_distance(point) <= safety_margin)

    def is_line_in_collision(self, p1: np.ndarray, p2: np.ndarray,
                             safety_margin: float = DEFAULT_SAFETY_MARGIN, step: float = None) -> bool:
        """True if any point of segment p1-p2 is within `safety_margin` of this obstacle.
        (`step` is accepted for backward compatibility and ignored: the check is exact up
        to SEGMENT_CHECK_TOL, see Environment3D.is_segment_valid.)"""
        env = Environment3D(bounds=((-np.inf, np.inf),) * 3)
        env.add_obstacle(self)
        return not env.is_segment_valid(p1, p2, safety_margin)


class Environment3D:
    """Axis-aligned workspace bounds plus a list of obstacles."""

    def __init__(self, bounds, safety_margin: float = DEFAULT_SAFETY_MARGIN):
        """bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))."""
        self.bounds = tuple(tuple(float(v) for v in b) for b in bounds)
        self.safety_margin = float(safety_margin)
        self.obstacles: list[Obstacle3D] = []

    def add_obstacle(self, obstacle: Obstacle3D):
        self.obstacles.append(obstacle)

    # ------------------------------------------------------------------ queries
    def clearance(self, points) -> np.ndarray:
        """Distance from point(s) to the nearest obstacle surface (inf if no obstacles)."""
        P = np.asarray(points, dtype=np.float64)
        single = P.ndim == 1
        P = np.atleast_2d(P)
        d = np.full(len(P), np.inf)
        for obs in self.obstacles:
            d = np.minimum(d, obs.signed_distance(P))
        return float(d[0]) if single else d

    def in_bounds(self, points) -> np.ndarray:
        P = np.atleast_2d(np.asarray(points, dtype=np.float64))
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        ok = np.all((P >= lo) & (P <= hi), axis=1)
        return bool(ok[0]) if np.asarray(points).ndim == 1 else ok

    def is_point_valid(self, p: np.ndarray, safety_margin: float = None) -> bool:
        """In bounds and clearance strictly greater than the safety margin."""
        m = self.safety_margin if safety_margin is None else safety_margin
        p = np.asarray(p, dtype=np.float64)
        return bool(self.in_bounds(p) and self.clearance(p) > m)

    def is_segment_valid(self, p1: np.ndarray, p2: np.ndarray, safety_margin: float = None,
                         step: float = None) -> bool:
        """True if every point of the straight segment p1-p2 is valid.

        Lipschitz bisection: the clearance function is 1-Lipschitz, so if the clearance at
        the midpoint m of a sub-segment of length L exceeds margin + L/2, the whole
        sub-segment is valid.  If clearance(m) <= margin the segment is invalid.  Otherwise
        the sub-segment is split.  Sub-segments shorter than SEGMENT_CHECK_TOL are accepted
        when clearance(m) > margin, so the guarantee is clearance > margin - SEGMENT_CHECK_TOL/2
        everywhere on an accepted segment.  (`step` is accepted for backward compatibility.)
        Bounds are convex, so checking both endpoints covers the bounds constraint.
        """
        m = self.safety_margin if safety_margin is None else safety_margin
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        if not (self.in_bounds(a) and self.in_bounds(b)):
            return False
        if not self.obstacles:
            return True
        A, B = a[None, :], b[None, :]
        for _ in range(64):
            mid = 0.5 * (A + B)
            half = 0.5 * np.linalg.norm(B - A, axis=1)
            c = self.clearance(mid)
            if np.any(c <= m):
                return False
            unresolved = c <= m + half
            small = 2.0 * half < SEGMENT_CHECK_TOL
            split = unresolved & ~small
            if not np.any(split):
                return True
            A_s, B_s, M_s = A[split], B[split], mid[split]
            A = np.vstack([A_s, M_s])
            B = np.vstack([M_s, B_s])
        return False  # unreachable in practice; conservative

    def segment_min_clearance(self, p1, p2, spacing: float = 0.01) -> float:
        """Sampled minimum clearance along a segment (for reporting)."""
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / spacing)) + 1)
        pts = p1 + np.linspace(0.0, 1.0, n)[:, None] * (p2 - p1)
        return float(np.min(self.clearance(pts)))

    def path_min_clearance(self, path: np.ndarray, spacing: float = 0.01) -> float:
        path = np.asarray(path, float)
        if len(path) == 0:
            return float("nan")
        if len(path) == 1:
            return float(self.clearance(path[0]))
        return min(self.segment_min_clearance(path[i], path[i + 1], spacing) for i in range(len(path) - 1))

    def is_path_valid(self, path: np.ndarray, safety_margin: float = None) -> bool:
        path = np.asarray(path, float)
        if len(path) < 2:
            return False
        return all(self.is_segment_valid(path[i], path[i + 1], safety_margin) for i in range(len(path) - 1))

    def free_volume_upper_bound(self) -> float:
        """Volume of the bounding box (an upper bound on the free-space volume)."""
        return float(np.prod([hi - lo for lo, hi in self.bounds]))


def path_length(path: np.ndarray) -> float:
    path = np.asarray(path, float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def shortcut_path(env: Environment3D, path: np.ndarray, safety_margin: float) -> np.ndarray:
    """Greedy line-of-sight shortcutting.

    From the current waypoint, jump to the FURTHEST later waypoint that is connected by a
    valid segment (same margin as the planner).  The fallback is the next waypoint, which is
    only safe because the input path is required to be valid; this is asserted.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) <= 2:
        return path.copy()
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        nxt = None
        for j in range(len(path) - 1, i, -1):
            if env.is_segment_valid(path[i], path[j], safety_margin):
                nxt = j
                break
        if nxt is None:
            raise ValueError("shortcut_path: input path contains an invalid segment "
                             f"between waypoints {i} and {i + 1}")
        out.append(path[nxt])
        i = nxt
    return np.array(out, dtype=np.float64)


def _check_endpoints(env, start, goal, margin):
    if not env.is_point_valid(start, margin):
        return STATUS_INVALID_START
    if not env.is_point_valid(goal, margin):
        return STATUS_INVALID_GOAL
    return None


# =====================================================================
# 1. 3D A* Path Planner
# =====================================================================
class AStarPlanner3D:
    """A* over a voxel lattice (26-connectivity) + line-of-sight shortcutting.

    Nodes are lattice points x_min + res * (i, j, k).  A node is valid if it is in bounds
    with clearance > margin.  An edge between two valid nodes is valid if the straight
    segment is valid; when min(clearance(u), clearance(v)) > margin + |u - v| / 2 the edge
    is valid without further checks (Lipschitz argument), otherwise the full segment check
    runs.  The start and goal (which need not lie on the lattice) are connected to the
    lattice by checked segments.  Edge cost is Euclidean length; the heuristic is the
    Euclidean distance to the goal, which is admissible and consistent for these costs.
    """

    def __init__(self, env: Environment3D, resolution: float = 0.5, safety_margin: float = None,
                 max_expansions: int = 300_000, try_direct: bool = True):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self.env = env
        self.res = float(resolution)
        self.margin = env.safety_margin if safety_margin is None else float(safety_margin)
        self.max_expansions = int(max_expansions)
        self.try_direct = bool(try_direct)
        lo = np.array([b[0] for b in env.bounds])
        hi = np.array([b[1] for b in env.bounds])
        self._origin = lo
        self._n = np.floor((hi - lo) / self.res + 1e-9).astype(int) + 1   # lattice points per axis
        self._clear_cache: dict[tuple, float] = {}

    def _to_grid(self, p: np.ndarray) -> tuple[int, int, int]:
        idx = np.round((np.asarray(p, float) - self._origin) / self.res).astype(int)
        idx = np.clip(idx, 0, self._n - 1)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def _to_world(self, idx) -> np.ndarray:
        return self._origin + np.array(idx, dtype=np.float64) * self.res

    def _node_clearance(self, idx) -> float:
        """Clearance of a lattice node (-inf if outside the lattice)."""
        c = self._clear_cache.get(idx)
        if c is None:
            if any(i < 0 or i >= n for i, n in zip(idx, self._n)):
                c = -np.inf
            else:
                c = float(self.env.clearance(self._to_world(idx)))
            self._clear_cache[idx] = c
        return c

    def _edge_valid(self, u, v, length) -> bool:
        cu, cv = self._node_clearance(u), self._node_clearance(v)
        if min(cu, cv) > self.margin + 0.5 * length:
            return True
        return self.env.is_segment_valid(self._to_world(u), self._to_world(v), self.margin)

    def _fail(self, status, t0, explored=0, reason=""):
        return _EMPTY_PATH.copy(), {
            "algorithm": "3D A*", "status": status, "success": False, "reason": reason,
            "planning_time_ms": (time.perf_counter() - t0) * 1e3, "path_length_m": float("nan"),
            "raw_path_length_m": float("nan"), "raw_waypoints": 0, "smoothed_waypoints": 0,
            "nodes_explored": explored, "safety_margin_m": self.margin, "min_clearance_m": float("nan"),
        }

    def plan(self, start: np.ndarray, goal: np.ndarray, shortcut: bool = True) -> tuple[np.ndarray, dict]:
        """Plans start -> goal.  Returns (path [N x 3] or empty, info dict)."""
        t0 = time.perf_counter()
        s = np.asarray(start, dtype=np.float64)
        g = np.asarray(goal, dtype=np.float64)
        bad = _check_endpoints(self.env, s, g, self.margin)
        if bad:
            return self._fail(bad, t0, reason=f"{bad.split('_')[1]} point is out of bounds or within the safety margin")

        explored = 0
        if self.try_direct and self.env.is_segment_valid(s, g, self.margin):
            raw = np.vstack([s, g])
        else:
            raw, explored = self._search(s, g)
            if raw is None:
                reason = "expansion limit reached" if explored >= self.max_expansions else "search space exhausted"
                return self._fail(STATUS_NO_PATH, t0, explored, reason)

        # every raw segment is valid by construction; verify (cheap) before shortcutting
        if not self.env.is_path_valid(raw, self.margin):
            raise AssertionError("A* produced an invalid raw path (internal error)")
        path = shortcut_path(self.env, raw, self.margin) if shortcut else raw
        info = {
            "algorithm": "3D A*", "status": STATUS_SUCCESS, "success": True, "reason": "",
            "planning_time_ms": (time.perf_counter() - t0) * 1e3,
            "path_length_m": path_length(path), "raw_path_length_m": path_length(raw),
            "raw_waypoints": len(raw), "smoothed_waypoints": len(path),
            "nodes_explored": explored, "safety_margin_m": self.margin,
            "min_clearance_m": self.env.path_min_clearance(path),
            "raw_path": raw,
        }
        return path, info

    def _connectors(self, p):
        """Lattice nodes around p that are valid and reachable from p by a valid segment."""
        base = self._to_grid(p)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    idx = (base[0] + dx, base[1] + dy, base[2] + dz)
                    if self._node_clearance(idx) <= self.margin:
                        continue
                    w = self._to_world(idx)
                    if self.env.is_segment_valid(p, w, self.margin):
                        out.append((idx, float(np.linalg.norm(w - p))))
        return out

    def _search(self, s, g):
        motions = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx or dy or dz:
                        motions.append((dx, dy, dz, self.res * float(np.sqrt(dx * dx + dy * dy + dz * dz))))

        start_links = self._connectors(s)
        goal_links = {idx: c for idx, c in self._connectors(g)}
        if not start_links or not goal_links:
            return None, 0

        h = lambda idx: float(np.linalg.norm(self._to_world(idx) - g))
        g_score: dict = {}
        came: dict = {}
        open_set = []
        for idx, c in start_links:
            if c < g_score.get(idx, np.inf):
                g_score[idx] = c
                came[idx] = None          # parent is the start point
                heapq.heappush(open_set, (c + h(idx), c, idx))

        closed = set()
        best_goal = (np.inf, None)       # (total cost, last lattice node)
        explored = 0
        while open_set:
            f, gc, cur = heapq.heappop(open_set)
            if cur in closed or gc > g_score.get(cur, np.inf):
                continue
            if f >= best_goal[0]:
                break                     # no open node can improve the best complete path
            closed.add(cur)
            explored += 1
            if explored >= self.max_expansions:
                break
            if cur in goal_links:
                tot = gc + goal_links[cur]
                if tot < best_goal[0]:
                    best_goal = (tot, cur)
            for dx, dy, dz, cost in motions:
                nb = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                if nb in closed or self._node_clearance(nb) <= self.margin:
                    continue
                ng = gc + cost
                if ng >= g_score.get(nb, np.inf):
                    continue
                if not self._edge_valid(cur, nb, cost):
                    continue
                g_score[nb] = ng
                came[nb] = cur
                heapq.heappush(open_set, (ng + h(nb), ng, nb))

        if best_goal[1] is None:
            return None, explored
        nodes = []
        cur = best_goal[1]
        while cur is not None:
            nodes.append(self._to_world(cur))
            cur = came[cur]
        nodes.reverse()
        return np.vstack([s, *nodes, g]), explored


# =====================================================================
# 2. 3D RRT* Path Planner
# =====================================================================
@dataclass
class RRTNode:
    """Kept for backward compatibility (the planner stores the tree in arrays)."""
    point: np.ndarray
    parent: int = -1
    cost: float = 0.0


def rrt_star_gamma_star(free_volume: float, dim: int = 3) -> float:
    """gamma* = 2 (1 + 1/d)^(1/d) (mu(X_free) / zeta_d)^(1/d)  (Karaman & Frazzoli 2011, Thm 38),
    zeta_d = volume of the unit d-ball."""
    zeta = np.pi ** (dim / 2) / _gamma_fn(dim / 2 + 1)
    return 2.0 * (1.0 + 1.0 / dim) ** (1.0 / dim) * (free_volume / zeta) ** (1.0 / dim)


class RRTStarPlanner3D:
    """RRT* in the continuous 3D workspace.

    Each iteration: sample (goal with probability goal_bias, else uniform in the bounds),
    find the nearest tree node, steer at most `step_size` (eta) towards the sample, reject
    if the segment is invalid, choose the lowest-cost valid parent among the near nodes,
    insert, then rewire near nodes through the new node when that lowers their cost.  A
    rewired node's whole subtree has its cost shifted by the same amount, so stored costs
    are always exact path costs from the root.

    Near radius: r_n = min(gamma * (ln n / n)^(1/3), radius_cap), with
    gamma = gamma_factor * gamma*, gamma* computed from the bounding-box volume (an upper
    bound on the free volume, so gamma > gamma* still holds), and radius_cap = step_size
    unless `search_radius` is given.

    Goal: every new node within step_size of the goal with a valid segment to it becomes a
    goal candidate.  The returned path uses the candidate with the lowest CURRENT cost
    (cost(node) + |node - goal|), evaluated after all rewiring.
    """

    def __init__(self, env: Environment3D, max_iter: int = 1200, step_size: float = 1.2,
                 goal_bias: float = 0.15, search_radius: float = None, safety_margin: float = None,
                 gamma_factor: float = 1.1, seed=None, shortcut: bool = False):
        if step_size <= 0 or max_iter < 1 or not (0.0 <= goal_bias <= 1.0):
            raise ValueError("invalid RRT* parameters")
        self.env = env
        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.goal_bias = float(goal_bias)
        self.radius_cap = self.step_size if search_radius is None else float(search_radius)
        self.margin = env.safety_margin if safety_margin is None else float(safety_margin)
        self.gamma = gamma_factor * rrt_star_gamma_star(env.free_volume_upper_bound(), 3)
        self.seed = seed
        self.shortcut = bool(shortcut)
        # tree of the last call to plan() (for tests and plots)
        self.tree_points = _EMPTY_PATH.copy()
        self.tree_parents = np.zeros(0, dtype=int)
        self.tree_costs = np.zeros(0)
        self.rewire_count = 0
        self.subtree_rewires = 0

    def near_radius(self, n: int) -> float:
        n = max(n, 2)
        return float(min(self.gamma * (np.log(n) / n) ** (1.0 / 3.0), self.radius_cap))

    def _fail(self, status, t0, reason, partial=None, dist=float("nan"), n_nodes=0):
        info = {
            "algorithm": "3D RRT*", "status": status, "success": False, "reason": reason,
            "planning_time_ms": (time.perf_counter() - t0) * 1e3, "path_length_m": float("nan"),
            "raw_path_length_m": float("nan"), "waypoints": 0, "tree_nodes": n_nodes,
            "iterations": self.max_iter, "rewires": self.rewire_count,
            "safety_margin_m": self.margin, "min_clearance_m": float("nan"),
            "partial_path": _EMPTY_PATH.copy() if partial is None else partial,
            "distance_to_goal_m": dist, "cost_history": [],
        }
        return _EMPTY_PATH.copy(), info

    def plan(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        s = np.asarray(start, dtype=np.float64)
        g = np.asarray(goal, dtype=np.float64)
        self.rewire_count = 0
        self.subtree_rewires = 0
        bad = _check_endpoints(self.env, s, g, self.margin)
        if bad:
            return self._fail(bad, t0, f"{bad.split('_')[1]} point is out of bounds or within the safety margin")

        rng = np.random.default_rng(self.seed)
        lo = np.array([b[0] for b in self.env.bounds])
        hi = np.array([b[1] for b in self.env.bounds])
        cap = self.max_iter + 1
        pts = np.zeros((cap, 3))
        par = np.full(cap, -1, dtype=int)
        cost = np.zeros(cap)
        children: list[list[int]] = [[] for _ in range(cap)]
        pts[0] = s
        n = 1
        goal_cands: set[int] = set()
        history = []

        for it in range(self.max_iter):
            q_rand = g.copy() if rng.random() < self.goal_bias else rng.uniform(lo, hi)
            d_all = np.linalg.norm(pts[:n] - q_rand, axis=1)
            i_near = int(np.argmin(d_all))
            v = q_rand - pts[i_near]
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                continue
            q_new = q_rand if dist <= self.step_size else pts[i_near] + v / dist * self.step_size
            if not self.env.is_segment_valid(pts[i_near], q_new, self.margin):
                continue

            r = self.near_radius(n + 1)
            d_new = np.linalg.norm(pts[:n] - q_new, axis=1)
            near = np.flatnonzero(d_new <= r)
            if i_near not in near:
                near = np.append(near, i_near)
            # choose parent: cheapest candidate first, lazily collision-checked
            cand = cost[near] + d_new[near]
            best_parent, best_cost = None, np.inf
            for k in np.argsort(cand):
                j = int(near[k])
                if j == i_near or self.env.is_segment_valid(pts[j], q_new, self.margin):
                    best_parent, best_cost = j, float(cand[k])
                    break
            new = n
            pts[new] = q_new
            par[new] = best_parent
            cost[new] = best_cost
            children[best_parent].append(new)
            n += 1

            # rewire
            for j in near:
                j = int(j)
                if j == best_parent:
                    continue
                c_via = best_cost + float(d_new[j])
                if c_via < cost[j] - 1e-12 and self.env.is_segment_valid(q_new, pts[j], self.margin):
                    old_parent = par[j]
                    children[old_parent].remove(j)
                    par[j] = new
                    children[new].append(j)
                    delta = c_via - cost[j]
                    if children[j]:
                        self.subtree_rewires += 1     # descendants exist and are updated below
                    queue = deque([j])            # shift the cost of the whole subtree
                    while queue:
                        u = queue.popleft()
                        cost[u] += delta
                        queue.extend(children[u])
                    self.rewire_count += 1

            if np.linalg.norm(q_new - g) <= self.step_size and self.env.is_segment_valid(q_new, g, self.margin):
                goal_cands.add(new)
            if (it + 1) % 50 == 0 or it == self.max_iter - 1:
                best = min((cost[i] + np.linalg.norm(pts[i] - g) for i in goal_cands), default=np.inf)
                history.append((it + 1, float(best)))

        self.tree_points, self.tree_parents, self.tree_costs = pts[:n].copy(), par[:n].copy(), cost[:n].copy()

        def branch(i):
            out = []
            while i != -1:
                out.append(pts[i])
                i = par[i]
            return np.array(out[::-1])

        if not goal_cands:
            closest = int(np.argmin(np.linalg.norm(pts[:n] - g, axis=1)))
            partial = branch(closest)
            dist_goal = float(np.linalg.norm(pts[closest] - g))
            status = STATUS_PARTIAL if n > 1 else STATUS_NO_PATH
            path, info = self._fail(status, t0, "tree did not reach the goal within max_iter",
                                    partial, dist_goal, n)
            info["cost_history"] = history
            return path, info

        best_i = min(goal_cands, key=lambda i: cost[i] + np.linalg.norm(pts[i] - g))
        raw = np.vstack([branch(best_i), g])
        if not self.env.is_path_valid(raw, self.margin):
            raise AssertionError("RRT* produced an invalid path (internal error)")
        path = shortcut_path(self.env, raw, self.margin) if self.shortcut else raw
        info = {
            "algorithm": "3D RRT*", "status": STATUS_SUCCESS, "success": True, "reason": "",
            "planning_time_ms": (time.perf_counter() - t0) * 1e3,
            "path_length_m": path_length(path), "raw_path_length_m": path_length(raw),
            "waypoints": len(path), "raw_waypoints": len(raw), "tree_nodes": n,
            "iterations": self.max_iter, "rewires": self.rewire_count,
            "goal_candidates": len(goal_cands), "safety_margin_m": self.margin,
            "min_clearance_m": self.env.path_min_clearance(path), "cost_history": history,
            "raw_path": raw, "partial_path": _EMPTY_PATH.copy(), "distance_to_goal_m": 0.0,
        }
        return path, info
