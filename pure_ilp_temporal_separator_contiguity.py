import bisect
import time
import logging
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional, Any

import gurobipy as gp
from gurobipy import GRB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PureILPTemporalSeparatorContiguity:
    """
    Pure ILP-based temporal separator solver for the contiguity version.
    """

    def __init__(
        self,
        temporal_edges: List[Tuple[int, int, int]],
        source: int,
        target: int,
        deadline: int,
        max_timestamp: int,
        enforce_contiguity: bool = True,
    ):
        self.temporal_edges = temporal_edges
        self.source = int(source)
        self.target = int(target)
        self.deadline = int(deadline)
        self.max_timestamp = int(max_timestamp)
        self.enforce_contiguity = bool(enforce_contiguity)

        self.vertices = set()
        self.temporal_graph = defaultdict(list)  # t -> [(u, v)]

        for u, v, t in temporal_edges:
            u, v, t = int(u), int(v), int(t)
            self.vertices.add(u)
            self.vertices.add(v)
            self.temporal_graph[t].append((u, v))

        self.vertices = sorted(self.vertices)

        logger.info(
            f"Initialized with {len(self.vertices)} vertices, {len(temporal_edges)} edges"
        )
        logger.info(
            f"Source: {self.source}, Target: {self.target}, "
            f"Deadline(travel-time): {self.deadline}, Max timestamp: {self.max_timestamp}, "
            f"enforce_contiguity: {self.enforce_contiguity}"
        )

    def find_temporal_paths(self, max_paths: Optional[int] = None) -> List[List[Tuple[int, int]]]:
        """
        Find temporal paths from source to target with:
          - strictly increasing times
          - travel-time constraint: trt(P) = t_last - t_first + 1 <= self.deadline

        Returns:
            list of paths, each path is a list of (vertex, timestamp) pairs.
            Convention: first item is (source, 0).
        """
        paths: List[List[Tuple[int, int]]] = []
        times_sorted = sorted(self.temporal_graph.keys())

        def dfs(
            current_vertex: int,
            current_time: int,
            t_first: Optional[int],
            path: List[Tuple[int, int]],
            visited_vt: Set[Tuple[int, int]],
        ):
            if max_paths is not None and len(paths) >= max_paths:
                return

            if current_vertex == self.target:
                if t_first is None:
                    return
                trt = current_time - t_first + 1
                if trt <= self.deadline:
                    paths.append(path.copy())
                return

            if t_first is not None:
                trt_now = current_time - t_first + 1
                if trt_now > self.deadline:
                    return

            idx = bisect.bisect_right(times_sorted, current_time)
            for next_time in times_sorted[idx:]:
                if t_first is not None:
                    trt_next = next_time - t_first + 1
                    if trt_next > self.deadline:
                        break

                for u, v in self.temporal_graph[next_time]:
                    if u != current_vertex:
                        continue
                    if (v, next_time) in visited_vt:
                        continue

                    new_t_first = next_time if t_first is None else t_first
                    if (next_time - new_t_first + 1) > self.deadline:
                        continue

                    path.append((v, next_time))
                    visited_vt.add((v, next_time))
                    dfs(v, next_time, new_t_first, path, visited_vt)
                    path.pop()
                    visited_vt.remove((v, next_time))

        initial_path = [(self.source, 0)]
        initial_visited = {(self.source, 0)}
        dfs(self.source, 0, None, initial_path, initial_visited)

        if max_paths is not None and len(paths) >= max_paths:
            logger.warning(
                f"Path enumeration stopped at limit {max_paths}. "
                f"This may lead to incomplete ILP constraints!"
            )

        logger.info(
            f"Found {len(paths)} temporal paths from {self.source} to {self.target} "
            f"(travel-time deadline d={self.deadline})"
        )
        return paths

    def solve_separator(
        self, time_limit: int = 300
    ) -> Tuple[Optional[Dict[Tuple[int, int], int]], float, Dict]:
        """
        Solve the ILP formulation for temporal separator.
        """
        start_time = time.time()

        paths = self.find_temporal_paths()
        if not paths:
            logger.warning("No temporal paths found (within travel-time deadline)")
            return None, float("inf"), {
                "status": "no_paths",
                "solve_time": time.time() - start_time,
                "num_paths": 0,
                "enforce_contiguity": self.enforce_contiguity,
            }

        model = gp.Model("temporal_separator_contiguity")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", int(time_limit))

        separator_vertices = [v for v in self.vertices if v != self.source and v != self.target]
        T = self.max_timestamp

        # Decision variables
        x: Dict[Tuple[int, int], Any] = {}
        for v in separator_vertices:
            for t in range(1, T + 1):
                x[v, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{t}")

        # Objective
        model.setObjective(
            gp.quicksum(x[v, t] for v in separator_vertices for t in range(1, T + 1)),
            GRB.MINIMIZE,
        )

        # Path separation constraints
        for i, path in enumerate(paths):
            vars_on_path = []
            for v, t in path:
                if t >= 1 and v in separator_vertices:
                    vars_on_path.append(x[v, t])
            if vars_on_path:
                model.addConstr(gp.quicksum(vars_on_path) >= 1, name=f"path_separation_{i}")

        # FAST contiguity constraints (O(T^2) per vertex):
        # If x[v,t1]=1 and x[v,t2]=1 with t2 >= t1+2, then x[v,t1+1]=1.
        # Applied for ALL pairs (t1,t2) implies no holes => one interval (possibly empty).
        if self.enforce_contiguity:
            contig_count = 0
            for v in separator_vertices:
                for t1 in range(1, T):
                    for t2 in range(t1 + 2, T + 1):
                        model.addConstr(
                            x[v, t1] + x[v, t2] - 1 <= x[v, t1 + 1],
                            name=f"contig_{v}_{t1}_{t2}",
                        )
                        contig_count += 1
            logger.info(f"Added {contig_count} contiguity constraints (FAST O(T^2))")

        logger.info(
            f"Created ILP with {len(x)} variables, {len(paths)} path constraints, "
            f"contiguity={'ON' if self.enforce_contiguity else 'OFF'}"
        )

        solve_start = time.time()
        model.optimize()
        solve_time = time.time() - solve_start

        stats = {
            "status": model.status,
            "solve_time": solve_time,
            "total_time": time.time() - start_time,
            "num_variables": len(x),
            "num_paths": len(paths),
            "num_constraints": model.NumConstrs,
            "enforce_contiguity": self.enforce_contiguity,
        }

        if model.status == GRB.OPTIMAL:
            separator: Dict[Tuple[int, int], int] = {}
            for v in separator_vertices:
                for t in range(1, T + 1):
                    if x[v, t].X > 0.5:
                        separator[(v, t)] = 1

            objective_value = float(model.objVal)
            stats["objective_value"] = objective_value
            logger.info(f"Optimal solution found: objective = {objective_value}")
            logger.info(f"Separator size: {len(separator)} vertex-time pairs")
            return separator, objective_value, stats

        if model.status == GRB.INFEASIBLE:
            logger.error("Model is infeasible")
            return None, float("inf"), stats

        if model.status == GRB.TIME_LIMIT:
            logger.warning("Time limit reached")
            if model.SolCount > 0:
                separator = {}
                for v in separator_vertices:
                    for t in range(1, T + 1):
                        if x[v, t].X > 0.5:
                            separator[(v, t)] = 1
                return separator, float(model.objVal), stats
            return None, float("inf"), stats

        logger.error(f"Unexpected solver status: {model.status}")
        return None, float("inf"), stats

    def verify_separator(self, separator: Dict[Tuple[int, int], int]) -> bool:
        """
        Verify separator feasibility and contiguity.
        """
        # Check path blocking
        paths = self.find_temporal_paths()
        for path in paths:
            if not any((v, t) in separator for (v, t) in path):
                logger.error(f"Path not blocked: {path}")
                return False

        # Check contiguity: each vertex's selected times must be consecutive
        if self.enforce_contiguity:
            vertex_times = defaultdict(list)
            for (v, t), val in separator.items():
                if val == 1:
                    vertex_times[v].append(t)

            for v, times in vertex_times.items():
                times = sorted(times)
                if not times:
                    continue
                for i in range(len(times) - 1):
                    if times[i + 1] != times[i] + 1:
                        logger.error(f"Contiguity violated for vertex {v}: times {times}")
                        return False

        logger.info("Separator verified successfully")
        return True

    def print_separator_summary(self, separator: Dict[Tuple[int, int], int], stats: Dict):
        if separator is None:
            print("No feasible separator found")
            return

        print("\n" + "=" * 60)
        print("TEMPORAL SEPARATOR SOLUTION SUMMARY (CONTIGUITY, FAST)")
        print("=" * 60)
        print(f"Problem: s={self.source} -> z={self.target}, travel-time deadline d={self.deadline}")
        print(f"Graph: {len(self.vertices)} vertices, {len(self.temporal_edges)} temporal edges")
        print(f"Time horizon: 1 to {self.max_timestamp}")
        print(f"Contiguity constraints: {'ON' if self.enforce_contiguity else 'OFF'}")

        print("\nSolver Statistics:")
        print(f"  Status: {stats.get('status', 'unknown')}")
        print(f"  Solve time: {stats.get('solve_time', 0):.3f}s")
        print(f"  Total time: {stats.get('total_time', 0):.3f}s")
        print(f"  Variables: {stats.get('num_variables', 0)}")
        print(f"  Constraints: {stats.get('num_constraints', 0)}")
        print(f"  Feasible paths found: {stats.get('num_paths', 0)}")

        print("\nSeparator Solution:")
        print(f"  Objective value: {stats.get('objective_value', 0)}")
        print(f"  Separator size: {len(separator)} vertex-time pairs")

        vertex_times = defaultdict(list)
        for (v, t), val in separator.items():
            if val == 1:
                vertex_times[v].append(t)

        print("\nVertex Times:")
        for v in sorted(vertex_times.keys()):
            times = sorted(vertex_times[v])
            print(f"  Vertex {v}: {times}")

        print("=" * 60)


def main():
    temporal_edges = [
        (1, 2, 10),
        (2, 3, 20),
        (1, 3, 50),
        (3, 4, 60),
    ]
    source = 1
    target = 4
    max_timestamp = 60
    deadline = 55

    print("Pure ILP Temporal Separator Solver (FAST)")
    print("=========================================================")

    solver = PureILPTemporalSeparatorContiguity(
        temporal_edges=temporal_edges,
        source=source,
        target=target,
        deadline=deadline,
        max_timestamp=max_timestamp,
        enforce_contiguity=True,
    )

    separator, objective_value, stats = solver.solve_separator(time_limit=60)
    solver.print_separator_summary(separator, stats)

    if separator is not None:
        ok = solver.verify_separator(separator)
        print(f"\nSeparator validation: {'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
