"""
3D Path Planning Algorithms for Aerial Robotics
===============================================
Course: Advanced Robotics
Authors: Group 1 (Charitha Sri, Rishika, Shreedhanvi, Maanya Sri)

This module implements two distinct 3D path-planning algorithms from first principles
using pure NumPy:
  1. 3D A* Search:
     - Discretized 3D voxel grid with 26-connectivity.
     - Admissible Euclidean distance heuristic.
     - Greedy line-of-sight path smoothing / shortcutting.
  2. 3D RRT* (Rapidly-exploring Random Tree Star):
     - Asymptotically optimal sampling-based planner in continuous 3D space.
     - Goal-biased sampling, steering, obstacle collision detection.
     - Ball-neighborhood search and branch rewiring for cost minimization.

Both planners support 3D geometric obstacles (cylinders, spheres, boxes) representing
forest trees, surveillance watchtowers, and buildings.
"""

import heapq
import time
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Obstacle3D:
    """3D Obstacle representation."""
    kind: str           # 'cylinder', 'sphere', or 'box'
    center: np.ndarray  # [x, y, z] center position
    radius: float = 1.0 # Radius for cylinder or sphere (m)
    height: float = 5.0 # Height for cylinder along z (m)
    size: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 5.0])) # [dx, dy, dz] for box

    def is_in_collision(self, point: np.ndarray, safety_margin: float = 0.5) -> bool:
        """Checks if a 3D point penetrates the obstacle + safety margin."""
        p = np.asarray(point, dtype=np.float64)
        c = self.center

        if self.kind == 'cylinder':
            # Cylinder aligned with Z axis from z = 0 to z = height
            d_xy = np.linalg.norm(p[0:2] - c[0:2])
            z_min = c[2] - safety_margin
            z_max = c[2] + self.height + safety_margin
            return (d_xy <= self.radius + safety_margin) and (z_min <= p[2] <= z_max)

        elif self.kind == 'sphere':
            dist = np.linalg.norm(p - c)
            return dist <= (self.radius + safety_margin)

        elif self.kind == 'box':
            half = 0.5 * self.size + safety_margin
            return bool(np.all(np.abs(p - c) <= half))

        return False

    def is_line_in_collision(self, p1: np.ndarray, p2: np.ndarray, safety_margin: float = 0.5, step: float = 0.2) -> bool:
        """Checks if line segment between p1 and p2 intersects obstacle."""
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-4:
            return self.is_in_collision(p1, safety_margin)

        num_samples = max(2, int(np.ceil(length / step)))
        alphas = np.linspace(0.0, 1.0, num_samples)
        for alpha in alphas:
            pt = p1 + alpha * vec
            if self.is_in_collision(pt, safety_margin):
                return True
        return False


class Environment3D:
    """3D Workspace containing obstacle obstacles and spatial boundaries."""

    def __init__(self, bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]):
        """
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        self.bounds = bounds
        self.obstacles: list[Obstacle3D] = []

    def add_obstacle(self, obstacle: Obstacle3D):
        self.obstacles.append(obstacle)

    def is_point_valid(self, p: np.ndarray, safety_margin: float = 0.4) -> bool:
        """Verifies point is within boundaries and outside all obstacles."""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.bounds
        if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max and z_min <= p[2] <= z_max):
            return False
        for obs in self.obstacles:
            if obs.is_in_collision(p, safety_margin):
                return False
        return True

    def is_segment_valid(self, p1: np.ndarray, p2: np.ndarray, safety_margin: float = 0.4, step: float = 0.25) -> bool:
        """Verifies straight line between p1 and p2 is collision-free."""
        for obs in self.obstacles:
            if obs.is_line_in_collision(p1, p2, safety_margin, step):
                return False
        return True


# =====================================================================
# 1. 3D A* Path Planner
# =====================================================================
class AStarPlanner3D:
    """
    3D A* Search over a voxel lattice with 26-connectivity.
    Features heuristic search, Euclidean distance metric, and line-of-sight path shortcutting.
    """

    def __init__(self, env: Environment3D, resolution: float = 0.5):
        self.env = env
        self.res = resolution

    def _to_grid(self, p: np.ndarray) -> tuple[int, int, int]:
        (x_min, _), (y_min, _), (z_min, _) = self.env.bounds
        idx = np.round((p - np.array([x_min, y_min, z_min])) / self.res).astype(int)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def _to_world(self, idx: tuple[int, int, int]) -> np.ndarray:
        (x_min, _), (y_min, _), (z_min, _) = self.env.bounds
        return np.array([x_min, y_min, z_min], dtype=np.float64) + np.array(idx, dtype=np.float64) * self.res

    def plan(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Executes 3D A* search from start to goal.
        Returns:
            (path_waypoints [N x 3], planning_metrics_dict)
        """
        t_start = time.perf_counter()

        start_pt = np.asarray(start, dtype=np.float64)
        goal_pt = np.asarray(goal, dtype=np.float64)

        start_idx = self._to_grid(start_pt)
        goal_idx = self._to_grid(goal_pt)

        # 26-connectivity neighborhood motions
        motions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    cost = np.sqrt(dx**2 + dy**2 + dz**2) * self.res
                    motions.append((dx, dy, dz, cost))

        # Open set: priority queue storing (f_score, g_score, grid_idx)
        open_set = []
        h_start = np.linalg.norm(start_pt - goal_pt)
        heapq.heappush(open_set, (h_start, 0.0, start_idx))

        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_scores: dict[tuple[int, int, int], float] = {start_idx: 0.0}
        closed_set = set()

        nodes_explored = 0
        found = False

        while open_set:
            _, g_curr, curr = heapq.heappop(open_set)

            if curr in closed_set:
                continue
            closed_set.add(curr)
            nodes_explored += 1

            curr_world = self._to_world(curr)
            if np.linalg.norm(curr_world - goal_pt) <= 1.2 * self.res:
                came_from[goal_idx] = curr
                found = True
                break

            for dx, dy, dz, d_cost in motions:
                neighbor = (curr[0] + dx, curr[1] + dy, curr[2] + dz)
                if neighbor in closed_set:
                    continue

                neighbor_world = self._to_world(neighbor)
                if not self.env.is_point_valid(neighbor_world):
                    continue

                tentative_g = g_curr + d_cost
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = curr
                    g_scores[neighbor] = tentative_g
                    h = np.linalg.norm(neighbor_world - goal_pt)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor))

        if not found and goal_idx not in came_from:
            # Fallback direct path
            raw_path = np.vstack([start_pt, goal_pt])
        else:
            # Reconstruct path
            path = [goal_pt]
            curr = goal_idx
            while curr in came_from:
                curr = came_from[curr]
                path.append(self._to_world(curr))
            path.append(start_pt)
            path.reverse()
            raw_path = np.array(path, dtype=np.float64)

        # Path shortcutting / smoothing
        smoothed_path = self._shortcut_path(raw_path)

        t_elapsed = time.perf_counter() - t_start
        path_length = float(np.sum(np.linalg.norm(np.diff(smoothed_path, axis=0), axis=1)))

        metrics = {
            "algorithm": "3D A*",
            "planning_time_ms": t_elapsed * 1000.0,
            "path_length_m": path_length,
            "raw_waypoints": len(raw_path),
            "smoothed_waypoints": len(smoothed_path),
            "nodes_explored": nodes_explored
        }
        return smoothed_path, metrics

    def _shortcut_path(self, path: np.ndarray) -> np.ndarray:
        """Greedy ray-casting shortcutting to eliminate jagged lattice steps."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        curr_idx = 0
        while curr_idx < len(path) - 1:
            furthest = curr_idx + 1
            for target_idx in range(len(path) - 1, curr_idx, -1):
                if self.env.is_segment_valid(path[curr_idx], path[target_idx], safety_margin=0.3):
                    furthest = target_idx
                    break
            smoothed.append(path[furthest])
            curr_idx = furthest

        return np.array(smoothed, dtype=np.float64)


# =====================================================================
# 2. 3D RRT* Path Planner
# =====================================================================
@dataclass
class RRTNode:
    point: np.ndarray
    parent: int = -1
    cost: float = 0.0


class RRTStarPlanner3D:
    """
    3D RRT* (Rapidly-exploring Random Tree Star) Planner.
    Features:
      - Continuous configuration space exploration.
      - Goal-biased uniform sampling.
      - Asymptotically optimal rewiring within ball radius r_n.
      - Obstacle avoidance for quadcopter flight corridors.
    """

    def __init__(
        self,
        env: Environment3D,
        max_iter: int = 1200,
        step_size: float = 1.2,
        goal_bias: float = 0.15,
        search_radius: float = 2.5
    ):
        self.env = env
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.search_radius = search_radius

    def plan(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Executes 3D RRT* search.
        Returns:
            (path_waypoints [N x 3], planning_metrics_dict)
        """
        t_start = time.perf_counter()

        start_pt = np.asarray(start, dtype=np.float64)
        goal_pt = np.asarray(goal, dtype=np.float64)

        tree = [RRTNode(point=start_pt, parent=-1, cost=0.0)]
        goal_indices = []

        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.env.bounds

        for i in range(self.max_iter):
            # 1. Sample with goal bias
            if np.random.rand() < self.goal_bias:
                q_rand = goal_pt.copy()
            else:
                q_rand = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                    np.random.uniform(z_min, z_max)
                ])

            # 2. Nearest neighbor
            tree_pts = np.array([node.point for node in tree])
            dists = np.linalg.norm(tree_pts - q_rand, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_pt = tree[nearest_idx].point

            # 3. Steer
            vec = q_rand - nearest_pt
            dist = np.linalg.norm(vec)
            if dist <= self.step_size:
                q_new = q_rand
            else:
                q_new = nearest_pt + (vec / dist) * self.step_size

            # 4. Collision check
            if not self.env.is_segment_valid(nearest_pt, q_new, safety_margin=0.35):
                continue

            # 5. Find near neighbors within search radius
            dists_to_new = np.linalg.norm(tree_pts - q_new, axis=1)
            near_indices = np.where(dists_to_new <= self.search_radius)[0]

            # 6. Choose best parent to minimize cost
            min_cost = tree[nearest_idx].cost + np.linalg.norm(q_new - nearest_pt)
            best_parent = nearest_idx

            for idx in near_indices:
                near_node = tree[idx]
                cost_candidate = near_node.cost + np.linalg.norm(q_new - near_node.point)
                if cost_candidate < min_cost:
                    if self.env.is_segment_valid(near_node.point, q_new, safety_margin=0.35):
                        min_cost = cost_candidate
                        best_parent = idx

            # 7. Add new node
            new_idx = len(tree)
            tree.append(RRTNode(point=q_new, parent=best_parent, cost=min_cost))

            # 8. Rewire tree in near neighborhood
            for idx in near_indices:
                if idx == best_parent:
                    continue
                near_node = tree[idx]
                potential_cost = min_cost + np.linalg.norm(q_new - near_node.point)
                if potential_cost < near_node.cost:
                    if self.env.is_segment_valid(q_new, near_node.point, safety_margin=0.35):
                        near_node.parent = new_idx
                        near_node.cost = potential_cost

            # 9. Check if close enough to connect directly to goal
            dist_to_goal = np.linalg.norm(q_new - goal_pt)
            if dist_to_goal <= self.step_size:
                if self.env.is_segment_valid(q_new, goal_pt, safety_margin=0.35):
                    goal_idx = len(tree)
                    tree.append(RRTNode(point=goal_pt, parent=new_idx, cost=min_cost + dist_to_goal))
                    goal_indices.append(goal_idx)

        # Extract best path
        if not goal_indices:
            # Connect to closest node
            tree_pts = np.array([node.point for node in tree])
            closest_idx = int(np.argmin(np.linalg.norm(tree_pts - goal_pt, axis=1)))
            path = [goal_pt]
            curr = closest_idx
        else:
            # Pick lowest cost goal node
            best_goal_idx = goal_indices[np.argmin([tree[idx].cost for idx in goal_indices])]
            path = []
            curr = best_goal_idx

        while curr != -1:
            path.append(tree[curr].point)
            curr = tree[curr].parent
        path.reverse()

        raw_path = np.array(path, dtype=np.float64)
        t_elapsed = time.perf_counter() - t_start
        path_length = float(np.sum(np.linalg.norm(np.diff(raw_path, axis=0), axis=1)))

        metrics = {
            "algorithm": "3D RRT*",
            "planning_time_ms": t_elapsed * 1000.0,
            "path_length_m": path_length,
            "waypoints": len(raw_path),
            "tree_nodes": len(tree),
            "iterations": self.max_iter
        }
        return raw_path, metrics
