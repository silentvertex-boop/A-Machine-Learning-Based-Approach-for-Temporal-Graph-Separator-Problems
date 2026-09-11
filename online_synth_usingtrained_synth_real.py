

import os
import re
import sys
import time
import random
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Tuple, Dict, Any, Set, Optional

import torch

# -----------------------------
# SETUP (EDIT HERE ONLY)
# -----------------------------
SYNTHETIC_INPUT = "/Users/mehdi/Desktop/Network_Analysis/TSD/python_v2/synthetic_graphs/ts50/synthetic_temporal_graph_munich.txt"

# JOINT MODEL
MODEL_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/ML_result/trained_model_joint_real_synthetic_contiguity"
BASE_OUT_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/ML_result/Synthetic"
MODEL_CODE_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/python_v2"

MAX_ITERS_PER_INSTANCE = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# ---- Beam-search controls ----
BEAM_WIDTH = 4
EXPAND_TOP_K = 4
PRINT_EVERY = 10

# ---- Move scoring ----
LAMBDA_DELTA = 0.03
LAMBDA_NEW_VERTEX = 0.20

# ---- State scoring ----
GAMMA_VERTICES = 0.50
ETA_MODEL = 0.20

# ---- Repair controls ----
ENABLE_REPAIR = True
REPAIR_MAX_ROUNDS = 20
# -----------------------------

if not os.path.exists(SYNTHETIC_INPUT):
    raise FileNotFoundError(f"SYNTHETIC_INPUT not found: {SYNTHETIC_INPUT}")

sys.setrecursionlimit(20000)

sys.path.append(MODEL_CODE_DIR)
from model_inductive_temporal import InductiveTemporalScorer  # noqa

INF_DIST = 10**9

_SYN_SOURCE_RE = re.compile(r"^\s*#\s*source\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_TARGET_RE = re.compile(r"^\s*#\s*target\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_DEADLINE_RE = re.compile(r"^\s*#\s*deadline\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_MAXT_RE = re.compile(r"^\s*#\s*max_timestamp\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_HORIZON_RE = re.compile(r"^\s*#\s*Max\s+timestamps\s*\(horizon\s*T\)\s*:\s*(\d+)\s*$", re.IGNORECASE)


def parse_name_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"synthetic_temporal_graph_(.+)$", stem, re.IGNORECASE)
    return m.group(1) if m else stem


# ============================================================================
# IO / graph utilities
# ============================================================================
def read_synthetic_uvt(path: str):
    edges: List[Tuple[int, int, int]] = []
    nodes: Set[int] = set()
    max_t_seen = 0

    source: Optional[int] = None
    target: Optional[int] = None
    deadline: Optional[int] = None
    max_timestamp_header: Optional[int] = None
    horizon_T: Optional[int] = None

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                m = _SYN_SOURCE_RE.match(s)
                if m:
                    source = int(m.group(1))
                    continue
                m = _SYN_TARGET_RE.match(s)
                if m:
                    target = int(m.group(1))
                    continue
                m = _SYN_DEADLINE_RE.match(s)
                if m:
                    deadline = int(m.group(1))
                    continue
                m = _SYN_MAXT_RE.match(s)
                if m:
                    max_timestamp_header = int(m.group(1))
                    continue
                m = _SYN_HORIZON_RE.match(s)
                if m:
                    horizon_T = int(m.group(1))
                    continue
                continue

            parts = s.split()
            if len(parts) < 3:
                continue

            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, t))
            nodes.add(u)
            nodes.add(v)
            max_t_seen = max(max_t_seen, t)

    if not edges:
        raise ValueError(f"No edges parsed from: {path}")
    if source is None or target is None or deadline is None:
        raise ValueError(
            f"Synthetic header missing source/target/deadline in {path} | "
            f"parsed source={source}, target={target}, deadline={deadline}"
        )

    Tmax = max_timestamp_header if max_timestamp_header is not None else (
        horizon_T if horizon_T is not None else max_t_seen
    )

    meta = {
        "input_file": path,
        "num_nodes": len(nodes),
        "nodes_set": nodes,
        "max_timestamp": int(Tmax),
        "max_timestamp_seen": int(max_t_seen),
        "temporal_edges": len(edges),
    }
    inst = {
        "source": int(source),
        "target": int(target),
        "deadline": int(deadline),
    }
    return edges, meta, inst


def build_temporal_graph(edges: List[Tuple[int, int, int]]):
    tg: Dict[int, List[Tuple[int, int]]] = {}
    times = set()
    for u, v, t in edges:
        tg.setdefault(int(t), []).append((int(u), int(v)))
        times.add(int(t))
    return tg, sorted(times)


def build_static_adj(edges_idx: List[Tuple[int, int, int]], n: int):
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]
    for u, v, _t in edges_idx:
        out_adj[u].append(v)
        in_adj[v].append(u)
    return out_adj, in_adj


def bfs_dist(adj: List[List[int]], src: int) -> List[int]:
    dist = [INF_DIST] * len(adj)
    dq = deque([src])
    dist[src] = 0
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if dist[y] == INF_DIST:
                dist[y] = dist[x] + 1
                dq.append(y)
    return dist


def build_time_indices_mapped(edges_idx: List[Tuple[int, int, int]], n: int, T: int, w: int):
    by_in = defaultdict(list)
    by_out = defaultdict(list)
    for u, v, tau in edges_idx:
        by_in[(v, tau)].append(u)
        by_out[(u, tau)].append(v)

    in_index = {}
    out_index = {}
    for v in range(n):
        for t_ref in range(1, T + 1):
            inc = []
            out = []
            for tau in range(max(1, t_ref - w), t_ref + 1):
                for u in by_in.get((v, tau), []):
                    inc.append((u, tau))
            for tau in range(t_ref, min(T, t_ref + w) + 1):
                for u2 in by_out.get((v, tau), []):
                    out.append((u2, tau))
            if inc:
                in_index[(v, t_ref)] = inc
            if out:
                out_index[(v, t_ref)] = out
    return in_index, out_index


def build_node_time_features(
    edges_idx: List[Tuple[int, int, int]],
    n: int,
    T: int,
    w: int,
    s_idx: int,
    z_idx: int,
) -> torch.Tensor:
    out_adj, in_adj = build_static_adj(edges_idx, n)
    dist_s = bfs_dist(out_adj, s_idx)
    dist_z = bfs_dist(in_adj, z_idx)

    in_deg = [len(in_adj[v]) for v in range(n)]
    out_deg = [len(out_adj[v]) for v in range(n)]

    by_time_in = defaultdict(list)
    by_time_out = defaultdict(list)
    for u, v, tau in edges_idx:
        by_time_in[(v, tau)].append(u)
        by_time_out[(u, tau)].append(v)

    F = 8
    x = torch.zeros((n, T + 1, F), dtype=torch.float32)

    for v in range(n):
        ds = dist_s[v] if dist_s[v] < INF_DIST else (T + 5)
        dz = dist_z[v] if dist_z[v] < INF_DIST else (T + 5)
        for t in range(1, T + 1):
            in_cnt = 0
            out_cnt = 0
            for tau in range(max(1, t - w), t + 1):
                in_cnt += len(by_time_in.get((v, tau), []))
            for tau in range(t, min(T, t + w) + 1):
                out_cnt += len(by_time_out.get((v, tau), []))

            x[v, t, 0] = float(in_cnt)
            x[v, t, 1] = float(out_cnt)
            x[v, t, 2] = float(in_deg[v])
            x[v, t, 3] = float(out_deg[v])
            x[v, t, 4] = float(ds)
            x[v, t, 5] = float(dz)
            x[v, t, 6] = float(t) / float(T)
            x[v, t, 7] = float(
                torch.log(torch.tensor((in_cnt + 1) / (out_cnt + 1), dtype=torch.float32))
            )

    return x


# ============================================================================
# Interval / verifier logic
# ============================================================================
def blocked_set_from_intervals(intervals: Dict[int, Tuple[int, int]]) -> Set[Tuple[int, int]]:
    blocked: Set[Tuple[int, int]] = set()
    for v, (L, R) in intervals.items():
        for t in range(L, R + 1):
            blocked.add((v, t))
    return blocked


def expanded_pairs_count(intervals: Dict[int, Tuple[int, int]]) -> int:
    return sum(R - L + 1 for (L, R) in intervals.values())


def verify_path_one(
    temporal_graph: Dict[int, List[Tuple[int, int]]],
    times_sorted: List[int],
    source: int,
    target: int,
    deadline: int,
    blocked_vt: Set[Tuple[int, int]],
    max_depth_guard: int = 30000,
) -> List[Tuple[int, int]]:
    found: List[List[Tuple[int, int]]] = []

    def dfs(
        cur_v: int,
        cur_t: int,
        t_first: Optional[int],
        path: List[Tuple[int, int]],
        visited: Set[Tuple[int, int]],
    ):
        if found:
            return
        if len(visited) > max_depth_guard:
            return

        if cur_v == target:
            if t_first is not None and (cur_t - t_first + 1) <= deadline:
                found.append(path.copy())
            return

        if t_first is not None and (cur_t - t_first + 1) > deadline:
            return

        for nt in times_sorted:
            if nt <= cur_t:
                continue
            if t_first is not None and (nt - t_first + 1) > deadline:
                break

            for u, v in temporal_graph.get(nt, []):
                if u != cur_v:
                    continue
                if (v, nt) in blocked_vt:
                    continue
                if (v, nt) in visited:
                    continue

                new_first = nt if t_first is None else t_first
                if (nt - new_first + 1) > deadline:
                    continue

                path.append((v, nt))
                visited.add((v, nt))
                dfs(v, nt, new_first, path, visited)
                visited.remove((v, nt))
                path.pop()
                if found:
                    return

    dfs(source, 0, None, [(source, 0)], {(source, 0)})
    return found[0] if found else []


def is_feasible_separator(
    intervals: Dict[int, Tuple[int, int]],
    tg: Dict[int, List[Tuple[int, int]]],
    times_sorted: List[int],
    s0: int,
    z0: int,
    d: int,
) -> bool:
    blocked = blocked_set_from_intervals(intervals)
    path = verify_path_one(tg, times_sorted, s0, z0, d, blocked)
    return len(path) == 0


def interval_move(intervals: Dict[int, Tuple[int, int]], v: int, t: int) -> Tuple[int, int, int, int]:
    if v not in intervals:
        return 0, 1, t, t
    L, R = intervals[v]
    old_len = R - L + 1
    new_L = min(L, t)
    new_R = max(R, t)
    new_len = new_R - new_L + 1
    return old_len, new_len, new_L, new_R


# ============================================================================
# Beam state
# ============================================================================
class BeamState:
    def __init__(self, intervals: Dict[int, Tuple[int, int]], accumulated_model_score: float = 0.0, steps: int = 0):
        self.intervals = intervals
        self.accumulated_model_score = float(accumulated_model_score)
        self.steps = int(steps)

    def copy(self) -> "BeamState":
        return BeamState(
            intervals=dict(self.intervals),
            accumulated_model_score=self.accumulated_model_score,
            steps=self.steps,
        )

    def num_vertices(self) -> int:
        return len(self.intervals)

    def expanded_pairs(self) -> int:
        return expanded_pairs_count(self.intervals)

    def state_value(self) -> float:
        return (
            self.expanded_pairs()
            + GAMMA_VERTICES * self.num_vertices()
            - ETA_MODEL * self.accumulated_model_score
        )

    def signature(self) -> Tuple[Tuple[int, int, int], ...]:
        return tuple(sorted((v, L, R) for v, (L, R) in self.intervals.items()))


# ============================================================================
# Model scoring
# ============================================================================
def score_candidates_with_model(
    model: InductiveTemporalScorer,
    graphs_data: List[Dict[str, Any]],
    cand: List[Tuple[int, int]],
    node2idx: Dict[int, int],
    s_idx: int,
    z_idx: int,
    d: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    cand_v = torch.tensor([node2idx[v] for (v, _t) in cand], dtype=torch.long, device=device)
    cand_t = torch.tensor([int(_t) for (_v, _t) in cand], dtype=torch.long, device=device)

    B = len(cand)
    batch_graph = torch.zeros((B,), dtype=torch.long, device=device)
    batch_s = torch.full((B,), int(s_idx), dtype=torch.long, device=device)
    batch_z = torch.full((B,), int(z_idx), dtype=torch.long, device=device)
    batch_d = torch.full((B,), int(d), dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(
            batch_graph=batch_graph,
            batch_v=cand_v,
            batch_t=cand_t,
            batch_s=batch_s,
            batch_z=batch_z,
            batch_d=batch_d,
            graphs_data=graphs_data,
            device=device,
        )
        probs = torch.sigmoid(logits).detach().cpu()

    return [{"v": v, "t": t, "prob": float(probs[j].item())} for j, (v, t) in enumerate(cand)]


def generate_children(state: BeamState, scored_rows: List[Dict[str, Any]]) -> List[BeamState]:
    enriched = []
    for row in scored_rows:
        v, t, prob = row["v"], row["t"], row["prob"]
        old_len, new_len, newL, newR = interval_move(state.intervals, v, t)
        delta_len = new_len - old_len
        is_new = 1 if v not in state.intervals else 0
        move_score = prob - LAMBDA_DELTA * delta_len - LAMBDA_NEW_VERTEX * is_new
        enriched.append({
            "v": v,
            "t": t,
            "prob": prob,
            "delta_len": delta_len,
            "is_new": is_new,
            "newL": newL,
            "newR": newR,
            "move_score": move_score,
        })

    enriched.sort(
        key=lambda r: (-r["move_score"], -r["prob"], r["delta_len"], r["is_new"], r["v"], r["t"])
    )

    children = []
    seen = set()
    for row in enriched[:EXPAND_TOP_K]:
        child = state.copy()
        child.intervals[row["v"]] = (row["newL"], row["newR"])
        child.accumulated_model_score += row["prob"]
        child.steps += 1
        sig = child.signature()
        if sig in seen:
            continue
        seen.add(sig)
        children.append(child)
    return children


# ============================================================================
# Repair v1
# ============================================================================
def repair_delete_vertices(intervals, tg, times_sorted, s0, z0, d):
    current = dict(intervals)
    deletions = 0
    vertices = sorted(
        current.keys(),
        key=lambda v: ((current[v][1] - current[v][0] + 1), v),
        reverse=True,
    )
    for v in vertices:
        if v not in current:
            continue
        trial = dict(current)
        del trial[v]
        if is_feasible_separator(trial, tg, times_sorted, s0, z0, d):
            current = trial
            deletions += 1
    return current, deletions


def repair_shrink_intervals(intervals, tg, times_sorted, s0, z0, d):
    current = dict(intervals)
    removed_pairs = 0
    vertices = sorted(
        current.keys(),
        key=lambda v: ((current[v][1] - current[v][0] + 1), v),
        reverse=True,
    )

    for v in vertices:
        if v not in current:
            continue

        changed = True
        while changed and v in current:
            changed = False
            L, R = current[v]
            if L < R:
                trial = dict(current)
                trial[v] = (L + 1, R)
                if is_feasible_separator(trial, tg, times_sorted, s0, z0, d):
                    current = trial
                    removed_pairs += 1
                    changed = True

        changed = True
        while changed and v in current:
            changed = False
            L, R = current[v]
            if L < R:
                trial = dict(current)
                trial[v] = (L, R - 1)
                if is_feasible_separator(trial, tg, times_sorted, s0, z0, d):
                    current = trial
                    removed_pairs += 1
                    changed = True

    return current, removed_pairs


def repair_intervals_v1(intervals, tg, times_sorted, s0, z0, d):
    current = dict(intervals)
    stats = {
        "rounds": 0,
        "vertex_deletions": 0,
        "endpoint_shrinks": 0,
        "before_pairs": expanded_pairs_count(current),
        "before_vertices": len(current),
        "after_pairs": expanded_pairs_count(current),
        "after_vertices": len(current),
    }

    for _ in range(REPAIR_MAX_ROUNDS):
        stats["rounds"] += 1
        improved = False

        current2, n_del = repair_delete_vertices(current, tg, times_sorted, s0, z0, d)
        if n_del > 0:
            current = current2
            stats["vertex_deletions"] += n_del
            improved = True

        current2, n_shrink = repair_shrink_intervals(current, tg, times_sorted, s0, z0, d)
        if n_shrink > 0:
            current = current2
            stats["endpoint_shrinks"] += n_shrink
            improved = True

        if not improved:
            break

    stats["after_pairs"] = expanded_pairs_count(current)
    stats["after_vertices"] = len(current)
    return current, stats


# ============================================================================
# Output
# ============================================================================
def write_result_txt(
    out_path: str,
    *,
    name: str,
    input_file: str,
    s0: int,
    z0: int,
    d: int,
    best_state: BeamState,
    repaired_intervals: Dict[int, Tuple[int, int]],
    repair_stats: Dict[str, int],
    total_wall: float,
    ml_wall: float,
    repair_wall: float,
    status: str,
    dataset_type: str,
):
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ML-GUIDED TEMPORAL SEPARATOR (SYNTHETIC, JOINT REAL+SYNTHETIC TRAINING, INTERVAL-AWARE BEAM SEARCH + REPAIR V1)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Name: {name}\n")
        f.write(f"Input file: {input_file}\n\n")

        f.write("INSTANCE\n")
        f.write(f"  source: {s0}\n")
        f.write(f"  target: {z0}\n")
        f.write(f"  deadline(travel_time): {d}\n\n")

        f.write("SOLVER\n")
        f.write("  method: ML-guided verify--cut (online), interval-aware beam search + exact repair v1\n")
        f.write(f"  checkpoint_dataset_type: {dataset_type}\n")
        f.write(f"  status: {status}\n")
        f.write(f"  successful_iterations_beam: {best_state.steps}\n")
        f.write(f"  wall_time_seconds: {total_wall:.6f}\n")
        f.write(f"  ml_wall_time_seconds: {ml_wall:.6f}\n")
        f.write(f"  repair_wall_time_seconds: {repair_wall:.6f}\n")
        f.write(f"  beam_width: {BEAM_WIDTH}\n")
        f.write(f"  expand_top_k: {EXPAND_TOP_K}\n")
        f.write(f"  lambda_delta: {LAMBDA_DELTA}\n")
        f.write(f"  lambda_new_vertex: {LAMBDA_NEW_VERTEX}\n")
        f.write(f"  gamma_vertices: {GAMMA_VERTICES}\n")
        f.write(f"  eta_model: {ETA_MODEL}\n")
        f.write(f"  enable_repair: {ENABLE_REPAIR}\n\n")

        f.write("BEAM SOLUTION BEFORE REPAIR\n")
        f.write(f"  vertices_in_separator: {best_state.num_vertices()}\n")
        f.write(f"  separator_size_pairs_expanded: {best_state.expanded_pairs()}\n")
        f.write(f"  accumulated_model_score: {best_state.accumulated_model_score:.6f}\n")
        f.write(f"  final_state_value: {best_state.state_value():.6f}\n\n")

        f.write("REPAIR\n")
        f.write(f"  rounds: {repair_stats['rounds']}\n")
        f.write(f"  vertex_deletions: {repair_stats['vertex_deletions']}\n")
        f.write(f"  endpoint_shrinks_total_pairs_removed: {repair_stats['endpoint_shrinks']}\n")
        f.write(f"  before_pairs: {repair_stats['before_pairs']}\n")
        f.write(f"  before_vertices: {repair_stats['before_vertices']}\n")
        f.write(f"  after_pairs: {repair_stats['after_pairs']}\n")
        f.write(f"  after_vertices: {repair_stats['after_vertices']}\n\n")

        f.write("FINAL SEPARATOR AFTER REPAIR\n")
        f.write(f"  vertices_in_separator: {len(repaired_intervals)}\n")
        f.write(f"  separator_size_pairs_expanded: {expanded_pairs_count(repaired_intervals)}\n\n")

        f.write("  intervals_per_vertex:\n")
        for v in sorted(repaired_intervals.keys()):
            L, R = repaired_intervals[v]
            f.write(f"    {v}: [{L}]\n" if L == R else f"    {v}: [{L},{R}]\n")


# ============================================================================
# Main
# ============================================================================
def main():
    random.seed(SEED)

    name = parse_name_from_filename(SYNTHETIC_INPUT)
    OUT_DIR = os.path.join(
        BASE_OUT_DIR,
        f"online_results_synthetic_{name}_joint_real_synthetic_interval_beam_repair"
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[SANITY]")
    print(f"  SYNTHETIC_INPUT={SYNTHETIC_INPUT}")
    print(f"  MODEL_DIR={MODEL_DIR}")
    print(f"  OUT_DIR={OUT_DIR}")
    print(f"  DEVICE={DEVICE}")
    print(f"  BEAM_WIDTH={BEAM_WIDTH}")
    print(f"  EXPAND_TOP_K={EXPAND_TOP_K}")
    print(f"  LAMBDA_DELTA={LAMBDA_DELTA}")
    print(f"  LAMBDA_NEW_VERTEX={LAMBDA_NEW_VERTEX}")
    print(f"  GAMMA_VERTICES={GAMMA_VERTICES}")
    print(f"  ETA_MODEL={ETA_MODEL}")
    print(f"  ENABLE_REPAIR={ENABLE_REPAIR}")
    print()

    ckpt_path = os.path.join(MODEL_DIR, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    for k in ("state_dict", "meta", "hparams"):
        if k not in ckpt:
            raise KeyError(f"Checkpoint missing key '{k}'. Found keys: {list(ckpt.keys())}")

    state_dict = ckpt["state_dict"]
    meta = ckpt["meta"]
    hparams = ckpt["hparams"]

    model_w = int(meta["window_w"])
    feat_dim = int(meta["feat_dim"])
    dataset_type = meta.get("dataset_type", "unknown")

    print(f"[INFO] Loaded checkpoint dataset_type={dataset_type}")
    if dataset_type != "joint_real_synthetic":
        print("[WARN] This checkpoint does not declare dataset_type='joint_real_synthetic'.")

    device = torch.device(DEVICE)
    model = InductiveTemporalScorer(
        feat_dim=feat_dim,
        window_w=model_w,
        delta_dim=hparams["delta_dim"],
        msg_dim=hparams["msg_dim"],
        hidden_dim=hparams["hidden_dim"],
        h_dim=hparams["h_dim"],
        dropout=hparams["dropout"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    edges, gmeta, inst = read_synthetic_uvt(SYNTHETIC_INPUT)
    s0 = int(inst["source"])
    z0 = int(inst["target"])
    d = int(inst["deadline"])

    tg, times_sorted = build_temporal_graph(edges)

    nodes = sorted(gmeta["nodes_set"])
    n = len(nodes)
    T = int(gmeta["max_timestamp"])
    node2idx = {v: i for i, v in enumerate(nodes)}
    edges_idx = [(node2idx[u], node2idx[v], t) for (u, v, t) in edges]

    in_index, out_index = build_time_indices_mapped(edges_idx, n, T, model_w)

    print("=" * 80)
    print("ONLINE ML-GUIDED SOLVER (SYNTHETIC, JOINT REAL+SYNTHETIC TRAINED MODEL)")
    print("=" * 80)
    print(f"Name: {name}")
    print(f"Input: {SYNTHETIC_INPUT}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Model window_w: {model_w} | feat_dim: {feat_dim}")
    print("=" * 80)

    t_total_0 = time.time()

    if s0 not in node2idx or z0 not in node2idx or s0 == z0 or d < 1:
        raise ValueError(f"Invalid synthetic instance: s={s0}, z={z0}, d={d}")

    s_idx = node2idx[s0]
    z_idx = node2idx[z0]

    x_tensor = build_node_time_features(edges_idx, n, T, model_w, s_idx, z_idx).to(device)
    graphs_data = [{
        "x_tensor": x_tensor,
        "in_index": in_index,
        "out_index": out_index,
        "T": T,
    }]

    start_state = BeamState(intervals={})
    beam: List[BeamState] = [start_state]
    best_finished: Optional[BeamState] = None
    status = "ok"

    t_ml_0 = time.time()

    for step in range(1, MAX_ITERS_PER_INSTANCE + 1):
        next_candidates: List[BeamState] = []
        seen_signatures = set()
        any_expanded = False
        all_finished = True

        for state in beam:
            blocked_vt = blocked_set_from_intervals(state.intervals)
            path = verify_path_one(tg, times_sorted, s0, z0, d, blocked_vt)

            if not path:
                if best_finished is None or state.state_value() < best_finished.state_value():
                    best_finished = state
                next_candidates.append(state)
                continue

            all_finished = False
            cand = [(v, t) for (v, t) in path if t >= 1 and v != s0 and v != z0]
            if not cand:
                next_candidates.append(state)
                continue

            any_expanded = True
            scored_rows = score_candidates_with_model(
                model=model,
                graphs_data=graphs_data,
                cand=cand,
                node2idx=node2idx,
                s_idx=s_idx,
                z_idx=z_idx,
                d=d,
                device=device,
            )

            children = generate_children(state=state, scored_rows=scored_rows)

            for child in children:
                sig = child.signature()
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                next_candidates.append(child)

        if not next_candidates:
            status = "stuck_no_states"
            break

        next_candidates.sort(
            key=lambda st: (st.state_value(), st.num_vertices(), st.expanded_pairs())
        )
        beam = next_candidates[:BEAM_WIDTH]

        if step % PRINT_EVERY == 0:
            leader = beam[0]
            print(
                f"  step={step:4d} | beam_size={len(beam)} | "
                f"leader_verts={leader.num_vertices():3d} | "
                f"leader_pairs={leader.expanded_pairs():4d} | "
                f"leader_value={leader.state_value():.4f}"
            )

        if all_finished:
            break
        if not any_expanded:
            status = "stuck_no_expansion"
            break

    if best_finished is not None:
        best_state = best_finished
    else:
        best_state = min(
            beam,
            key=lambda st: (st.state_value(), st.num_vertices(), st.expanded_pairs())
        )

    t_ml_1 = time.time()
    ml_wall = t_ml_1 - t_ml_0

    repaired_intervals = dict(best_state.intervals)
    repair_stats = {
        "rounds": 0,
        "vertex_deletions": 0,
        "endpoint_shrinks": 0,
        "before_pairs": expanded_pairs_count(repaired_intervals),
        "before_vertices": len(repaired_intervals),
        "after_pairs": expanded_pairs_count(repaired_intervals),
        "after_vertices": len(repaired_intervals),
    }

    repair_wall = 0.0
    if ENABLE_REPAIR:
        t_repair_0 = time.time()
        repaired_intervals, repair_stats = repair_intervals_v1(
            repaired_intervals, tg, times_sorted, s0, z0, d
        )
        t_repair_1 = time.time()
        repair_wall = t_repair_1 - t_repair_0

    t_total_1 = time.time()
    total_wall = t_total_1 - t_total_0

    if best_state.steps >= MAX_ITERS_PER_INSTANCE:
        status = "hit_iter_cap"

    out_txt = os.path.join(OUT_DIR, f"{name}_s{s0}_z{z0}_d{d}.txt")
    write_result_txt(
        out_path=out_txt,
        name=name,
        input_file=SYNTHETIC_INPUT,
        s0=s0,
        z0=z0,
        d=d,
        best_state=best_state,
        repaired_intervals=repaired_intervals,
        repair_stats=repair_stats,
        total_wall=total_wall,
        ml_wall=ml_wall,
        repair_wall=repair_wall,
        status=status,
        dataset_type=dataset_type,
    )

    print(
        f"[DONE] status={status} | "
        f"before_pairs={repair_stats['before_pairs']} -> after_pairs={repair_stats['after_pairs']} | "
        f"before_verts={repair_stats['before_vertices']} -> after_verts={repair_stats['after_vertices']} | "
        f"total={total_wall:.3f}s | ml={ml_wall:.3f}s | repair={repair_wall:.3f}s"
    )
    print(f"       saved: {out_txt}")

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("SUMMARY (Synthetic online ML-guided solver, joint real+synthetic training, interval-aware beam search + repair v1)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Name: {name}\n")
        f.write(f"Input: {SYNTHETIC_INPUT}\n")
        f.write(f"Model dir: {MODEL_DIR}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"OUT_DIR: {OUT_DIR}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Model window_w: {model_w}\n")
        f.write(f"beam_width: {BEAM_WIDTH}\n")
        f.write(f"expand_top_k: {EXPAND_TOP_K}\n")
        f.write(f"lambda_delta: {LAMBDA_DELTA}\n")
        f.write(f"lambda_new_vertex: {LAMBDA_NEW_VERTEX}\n")
        f.write(f"gamma_vertices: {GAMMA_VERTICES}\n")
        f.write(f"eta_model: {ETA_MODEL}\n")
        f.write(f"enable_repair: {ENABLE_REPAIR}\n")
        f.write(f"dataset_type: {dataset_type}\n\n")
        f.write(f"s={s0} z={z0} d={d} status={status}\n")
        f.write(f"verts_before={repair_stats['before_vertices']} pairs_before={repair_stats['before_pairs']}\n")
        f.write(f"verts_after={repair_stats['after_vertices']} pairs_after={repair_stats['after_pairs']}\n")
        f.write(f"success_iters={best_state.steps}\n")
        f.write(f"total_time_s={total_wall:.6f}\n")
        f.write(f"ml_time_s={ml_wall:.6f}\n")
        f.write(f"repair_time_s={repair_wall:.6f}\n")

    print("\n" + "=" * 80)
    print("[DONE] Synthetic instance processed.")
    print(f"Results directory: {OUT_DIR}")
    print(f"Summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
