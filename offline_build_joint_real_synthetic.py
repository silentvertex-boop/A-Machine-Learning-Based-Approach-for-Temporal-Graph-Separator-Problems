import os
import re
import sys
import json
import time
import math
import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, Set, Optional, Union

import torch

# =============================================================================
# SETUP (EDIT HERE ONLY)
# =============================================================================
PURE_ILP_SOLVER_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/python_v2/pure_ilp_solver"
sys.path.append(PURE_ILP_SOLVER_DIR)
from pure_ilp_temporal_separator_contiguity import PureILPTemporalSeparatorContiguity  # noqa

OUT_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/ML_result/joint_real_synthetic_dataset_contiguity"

WINDOW_W = 5

# ILP supervision
ILP_TIME_LIMIT: int = 60
ILP_MAX_PATHS: Union[None, int, float] = None   # None or NaN => no cap, else int>=1
ENFORCE_CONTIGUITY: bool = True

# Path sampling supervision
PATH_SAMPLES = 3000
PATH_MAX_STEPS = 20000
SAMPLER_TRIES_MULT = 10
DFS_RESTARTS_PER_PATH = 8

# Dataset composition
NEG_RATIO_RANDOM = 3
HARD_NEG_MULT = 50
SOFT_POS_MULT = 30
SOFT_POS_MIN_LABEL = 0.2
SOFT_POS_MAX_LABEL = 0.8

REAL_UVT_FILES = [
    ("grenoble",
     "/Users/mehdi/Desktop/Network_Analysis/TSD/data_new_v2/grenoble/network_temporal_day_uvt_first2h.txt"),
    ("berlin",
     "/Users/mehdi/Desktop/Network_Analysis/TSD/data_new_v2/berlin/network_temporal_day_uvt_first2h.txt"),
    ("luxembourg",
     "/Users/mehdi/Desktop/Network_Analysis/TSD/data_new_v2/luxembourg/network_temporal_day_uvt_first2h.txt"),
]

SYNTHETIC_UVT_FILES = [
    ("friedrichshain_ts50",
     "/Users/mehdi/Desktop/Network_Analysis/TSD/python_v2/synthetic_graphs/ts50/synthetic_temporal_graph_friedrichshain-center.txt"),
    ("barcelona_ts50",
     "/Users/mehdi/Desktop/Network_Analysis/TSD/python_v2/synthetic_graphs/ts50/synthetic_temporal_graph_barcelona.txt"),
    ("anaheim_ts50",
     "/Users/mehdi/Desktop/Network_Analysis/TSD/python_v2/synthetic_graphs/ts50/synthetic_temporal_graph_anaheim.txt"),
]

SEED = 0
# =============================================================================

_COMBO_RE = re.compile(
    r"combo\s*(\d+)\s*:\s*source\s*=\s*(\d+)\s+target\s*=\s*(\d+)\s+deadline\s*=\s*(\d+)",
    re.IGNORECASE,
)

_SYN_SOURCE_RE = re.compile(r"^\s*#\s*source\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_TARGET_RE = re.compile(r"^\s*#\s*target\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_DEADLINE_RE = re.compile(r"^\s*#\s*deadline\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_MAXT_RE = re.compile(r"^\s*#\s*max_timestamp\s*=\s*(\d+)\s*$", re.IGNORECASE)
_SYN_HORIZON_RE = re.compile(r"^\s*#\s*Max\s+timestamps\s*\(horizon\s*T\)\s*:\s*(\d+)\s*$", re.IGNORECASE)
_SYN_DEADLINE_NOTE_RE = re.compile(r"^\s*#\s*deadline_note\s*=\s*(.*)\s*$", re.IGNORECASE)

INF_DIST = 10**9
ExampleKey = Tuple[int, int, int, int, int, int]  # (g_id, s_idx, z_idx, d, v_idx, t)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------
def validate_config() -> None:
    if WINDOW_W < 1:
        raise ValueError("WINDOW_W must be >= 1")
    if ILP_TIME_LIMIT < 1:
        raise ValueError("ILP_TIME_LIMIT must be >= 1")
    if PATH_SAMPLES < 0:
        raise ValueError("PATH_SAMPLES must be >= 0")
    if PATH_MAX_STEPS < 1:
        raise ValueError("PATH_MAX_STEPS must be >= 1")
    if SAMPLER_TRIES_MULT < 1:
        raise ValueError("SAMPLER_TRIES_MULT must be >= 1")
    if DFS_RESTARTS_PER_PATH < 1:
        raise ValueError("DFS_RESTARTS_PER_PATH must be >= 1")
    if NEG_RATIO_RANDOM < 0 or HARD_NEG_MULT < 0 or SOFT_POS_MULT < 0:
        raise ValueError("NEG_RATIO_RANDOM/HARD_NEG_MULT/SOFT_POS_MULT must be >= 0")
    if not (0.0 <= SOFT_POS_MIN_LABEL <= 1.0 and 0.0 <= SOFT_POS_MAX_LABEL <= 1.0):
        raise ValueError("SOFT_POS_MIN_LABEL and SOFT_POS_MAX_LABEL must be in [0,1]")
    if SOFT_POS_MIN_LABEL >= SOFT_POS_MAX_LABEL:
        raise ValueError("Require SOFT_POS_MIN_LABEL < SOFT_POS_MAX_LABEL")
    if not REAL_UVT_FILES and not SYNTHETIC_UVT_FILES:
        raise ValueError("Both REAL_UVT_FILES and SYNTHETIC_UVT_FILES are empty")


# -----------------------------------------------------------------------------
# ILP_MAX_PATHS cap normalization
# -----------------------------------------------------------------------------
def _is_no_cap(x: Union[None, int, float]) -> bool:
    return (x is None) or (isinstance(x, float) and math.isnan(x))


def _normalize_cap(x: Union[None, int, float]) -> Optional[int]:
    if _is_no_cap(x):
        return None
    if isinstance(x, bool):
        raise ValueError(f"ILP_MAX_PATHS must be None/NaN/int>=1, got bool {x!r}")
    if isinstance(x, int):
        if x < 1:
            raise ValueError(f"ILP_MAX_PATHS must be >= 1, got {x}")
        return x
    if isinstance(x, float):
        if not x.is_integer():
            raise ValueError(f"ILP_MAX_PATHS float must be an integer value, got {x}")
        xi = int(x)
        if xi < 1:
            raise ValueError(f"ILP_MAX_PATHS must be >= 1, got {xi}")
        return xi
    raise ValueError(f"ILP_MAX_PATHS must be None/NaN/int>=1, got {type(x)}")


# -----------------------------------------------------------------------------
# Parsing: real files with combos
# -----------------------------------------------------------------------------
def read_uvt_temporal_graph_with_combos(path: str):
    edges: List[Tuple[int, int, int]] = []
    nodes: Set[int] = set()
    max_t = 0
    combos: List[Dict[str, Any]] = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                m = _COMBO_RE.search(s)
                if m:
                    combos.append({
                        "combo_id": int(m.group(1)),
                        "source": int(m.group(2)),
                        "target": int(m.group(3)),
                        "deadline": int(m.group(4)),
                        "raw_line": s,
                    })
                continue

            parts = s.split()
            if len(parts) < 3:
                continue

            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, t))
            nodes.add(u)
            nodes.add(v)
            if t > max_t:
                max_t = t

    combos = sorted(combos, key=lambda d: d["combo_id"])
    meta = {
        "nodes_set": nodes,
        "max_timestamp": max_t,
        "temporal_edges": len(edges),
        "num_nodes": len(nodes),
    }
    return edges, meta, combos


# -----------------------------------------------------------------------------
# Parsing: synthetic single-instance files
# -----------------------------------------------------------------------------
def read_synthetic_uvt_single_instance(path: str):
    edges: List[Tuple[int, int, int]] = []
    nodes: Set[int] = set()
    max_t_seen = 0

    source: Optional[int] = None
    target: Optional[int] = None
    deadline: Optional[int] = None
    max_timestamp_header: Optional[int] = None
    horizon_T: Optional[int] = None
    deadline_note: Optional[str] = None

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
                m = _SYN_DEADLINE_NOTE_RE.match(s)
                if m:
                    deadline_note = m.group(1).strip()
                    continue
                continue

            parts = s.split()
            if len(parts) < 3:
                continue

            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, t))
            nodes.add(u)
            nodes.add(v)
            if t > max_t_seen:
                max_t_seen = t

    if not edges:
        raise ValueError(f"No edges parsed from: {path}")
    if source is None or target is None or deadline is None:
        raise ValueError(
            f"Synthetic header missing source/target/deadline in file: {path} | "
            f"parsed source={source}, target={target}, deadline={deadline}"
        )

    Tmax = max_timestamp_header if max_timestamp_header is not None else (
        horizon_T if horizon_T is not None else max_t_seen
    )

    meta = {
        "nodes_set": nodes,
        "max_timestamp": int(Tmax),
        "max_timestamp_seen": int(max_t_seen),
        "temporal_edges": len(edges),
        "num_nodes": len(nodes),
        "deadline_note": deadline_note,
    }
    instance = {
        "source": int(source),
        "target": int(target),
        "deadline": int(deadline),
    }
    return edges, meta, instance


# -----------------------------------------------------------------------------
# ILP runner
# -----------------------------------------------------------------------------
def cap_ilp_paths(solver: PureILPTemporalSeparatorContiguity, max_paths: Optional[int]) -> None:
    if max_paths is None:
        return
    orig = solver.find_temporal_paths

    def limited_find(max_paths_override=None):
        return orig(max_paths=max_paths)

    solver.find_temporal_paths = limited_find  # type: ignore


def run_ilp(
    edges: List[Tuple[int, int, int]],
    s: int,
    z: int,
    d: int,
    T: int,
    time_limit: int,
    max_paths: Optional[int],
    enforce_contiguity: bool,
) -> Dict[Tuple[int, int], int]:
    solver = PureILPTemporalSeparatorContiguity(
        edges, s, z, d, T,
        enforce_contiguity=enforce_contiguity
    )
    cap_ilp_paths(solver, max_paths)
    sep, _obj, _stats = solver.solve_separator(time_limit=time_limit)
    return sep if sep else {}


# -----------------------------------------------------------------------------
# Feature building
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Path sampling
# -----------------------------------------------------------------------------
def build_temporal_graph_mapped(edges_idx: List[Tuple[int, int, int]]):
    tg_by_out = defaultdict(list)  # (u,tau) -> [v]
    times = set()
    for u, v, tau in edges_idx:
        tg_by_out[(int(u), int(tau))].append(int(v))
        times.add(int(tau))
    return tg_by_out, sorted(times)


def sample_path_randomized_dfs(
    tg_by_out: Dict[Tuple[int, int], List[int]],
    times_sorted: List[int],
    s_idx: int,
    z_idx: int,
    d: int,
    rng: random.Random,
    max_steps_guard: int,
    restarts: int,
) -> List[Tuple[int, int]]:
    for _ in range(restarts):
        path = [(s_idx, 0)]
        visited = {(s_idx, 0)}
        cur = s_idx
        cur_t = 0
        t_first = None
        steps = 0

        stack: List[Tuple[int, int, Optional[int], List[Tuple[int, int, Optional[int]]]]] = []

        while steps < max_steps_guard:
            steps += 1

            if cur == z_idx:
                if t_first is not None and (cur_t - t_first + 1) <= d:
                    return path
                break

            moves: List[Tuple[int, int, Optional[int]]] = []
            for tau in times_sorted:
                if tau <= cur_t:
                    continue
                if t_first is not None and (tau - t_first + 1) > d:
                    break
                for vv in tg_by_out.get((cur, tau), []):
                    if (vv, tau) in visited:
                        continue
                    new_first = tau if t_first is None else t_first
                    if (tau - new_first + 1) > d:
                        continue
                    moves.append((vv, tau, new_first))

            if moves:
                rng.shuffle(moves)
                first = moves.pop()
                stack.append((cur, cur_t, t_first, moves))

                vv, tau, new_first = first
                path.append((vv, tau))
                visited.add((vv, tau))
                cur, cur_t, t_first = vv, tau, new_first
                continue

            while stack and not stack[-1][3]:
                stack.pop()
                if len(path) > 1:
                    visited.remove(path[-1])
                    path.pop()

            if not stack:
                break

            cur, cur_t, t_first, rem = stack[-1]
            vv, tau, new_first = rem.pop()

            while path and (path[-1][0] != cur or path[-1][1] != cur_t):
                if len(path) <= 1:
                    break
                visited.remove(path[-1])
                path.pop()

            path.append((vv, tau))
            visited.add((vv, tau))
            cur, cur_t, t_first = vv, tau, new_first

    return []


def collect_path_frequencies(
    tg_by_out: Dict[Tuple[int, int], List[int]],
    times_sorted: List[int],
    s_idx: int,
    z_idx: int,
    d: int,
    n_samples: int,
    rng: random.Random,
) -> Tuple[Dict[Tuple[int, int], int], int, int]:
    counts = defaultdict(int)
    got = 0
    tries = 0
    max_tries = n_samples * SAMPLER_TRIES_MULT

    while got < n_samples and tries < max_tries:
        tries += 1
        p = sample_path_randomized_dfs(
            tg_by_out=tg_by_out,
            times_sorted=times_sorted,
            s_idx=s_idx,
            z_idx=z_idx,
            d=d,
            rng=rng,
            max_steps_guard=PATH_MAX_STEPS,
            restarts=DFS_RESTARTS_PER_PATH,
        )
        if not p:
            continue
        got += 1
        for (v, t) in p:
            if t >= 1:
                counts[(v, t)] += 1

    return dict(counts), got, tries


# -----------------------------------------------------------------------------
# Label mixing / priority rule
# -----------------------------------------------------------------------------
def add_labeled_example(
    store: Dict[ExampleKey, float],
    key: ExampleKey,
    y: float,
) -> None:
    y = float(y)
    if y < 0.0 or y > 1.0:
        raise ValueError(f"Label y must be in [0,1], got {y}")

    if key not in store:
        store[key] = y
        return

    old = float(store[key])

    if old >= 1.0 - 1e-12:
        return
    if y >= 1.0 - 1e-12:
        store[key] = 1.0
        return

    if 0.0 < old < 1.0 and 0.0 < y < 1.0:
        if y > old:
            store[key] = y
        return

    if old == 0.0 and 0.0 < y < 1.0:
        store[key] = y
        return


# -----------------------------------------------------------------------------
# Core processor for one query
# -----------------------------------------------------------------------------
def process_one_query(
    *,
    g_id: int,
    prefix: str,
    graph_block: Dict[str, Any],
    edges: List[Tuple[int, int, int]],
    edges_idx: List[Tuple[int, int, int]],
    tg_by_out: Dict[Tuple[int, int], List[int]],
    times_sorted: List[int],
    T: int,
    n: int,
    s0: int,
    z0: int,
    d: int,
    node2idx: Dict[int, int],
    examples_store: Dict[ExampleKey, float],
    max_paths_cap: Optional[int],
    rng: random.Random,
) -> None:
    if s0 == z0:
        log(f"  {prefix} SKIP (s==z)")
        return
    if s0 not in node2idx or z0 not in node2idx:
        log(f"  {prefix} SKIP (node missing)")
        return
    if d < 1:
        log(f"  {prefix} SKIP (d<1)")
        return

    s_idx = node2idx[s0]
    z_idx = node2idx[z0]

    log(f"  {prefix} s={s0} z={z0} d={d}")

    # ---------- ILP supervision ----------
    t0 = time.time()
    sep = run_ilp(
        edges=edges,
        s=s0,
        z=z0,
        d=d,
        T=T,
        time_limit=ILP_TIME_LIMIT,
        max_paths=max_paths_cap,
        enforce_contiguity=ENFORCE_CONTIGUITY,
    )
    t_ilp = time.time() - t0

    positives_orig = [(v, t) for (v, t) in sep.keys() if v != s0 and v != z0 and t >= 1]
    log(f"    {prefix} ILP sep_pairs={len(positives_orig)} time={t_ilp:.3f}s")

    if not positives_orig:
        return

    pos_pairs = [(node2idx[v], int(t)) for (v, t) in positives_orig]
    pos_set = set(pos_pairs)

    # cache features for this (s,z)
    cache_key_sz = f"{s_idx}_{z_idx}"
    if cache_key_sz not in graph_block["x_cache"]:
        graph_block["x_cache"][cache_key_sz] = build_node_time_features(
            edges_idx, n, T, WINDOW_W, s_idx, z_idx
        )

    # ---------- Path sampling supervision ----------
    t1 = time.time()
    counts, got, tries = collect_path_frequencies(
        tg_by_out=tg_by_out,
        times_sorted=times_sorted,
        s_idx=s_idx,
        z_idx=z_idx,
        d=d,
        n_samples=PATH_SAMPLES,
        rng=rng,
    )
    t_paths = time.time() - t1
    max_c = max(counts.values()) if counts else 0
    log(
        f"    {prefix} PATH sampling: got={got}/{PATH_SAMPLES} tries={tries} | "
        f"unique(v,t)={len(counts)} | max_count={max_c} | time={t_paths:.3f}s"
    )
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    # ---------- Add ILP positives ----------
    for (v_idx, tt) in pos_pairs:
        key: ExampleKey = (g_id, s_idx, z_idx, int(d), int(v_idx), int(tt))
        add_labeled_example(examples_store, key, 1.0)

    # ---------- Soft positives ----------
    K_soft = max(0, SOFT_POS_MULT * len(pos_pairs))
    soft_added = 0
    for (vt, ccount) in ranked:
        if soft_added >= K_soft:
            break
        if vt in pos_set:
            continue
        v_idx, tt = vt
        frac = ccount / max(1, max_c)
        y_soft = SOFT_POS_MIN_LABEL + (SOFT_POS_MAX_LABEL - SOFT_POS_MIN_LABEL) * frac
        key = (g_id, s_idx, z_idx, int(d), int(v_idx), int(tt))
        add_labeled_example(examples_store, key, float(y_soft))
        soft_added += 1

    # ---------- Hard negatives ----------
    K_hard = max(0, HARD_NEG_MULT * len(pos_pairs))
    hard_added = 0
    for (vt, _ccount) in ranked:
        if hard_added >= K_hard:
            break
        if vt in pos_set:
            continue
        v_idx, tt = vt
        key = (g_id, s_idx, z_idx, int(d), int(v_idx), int(tt))
        add_labeled_example(examples_store, key, 0.0)
        hard_added += 1

    # ---------- Random negatives ----------
    K_rand = NEG_RATIO_RANDOM * len(pos_pairs)
    rand_added = 0
    max_rand_tries = max(10_000, 50 * K_rand)
    rand_tries = 0

    while rand_added < K_rand and rand_tries < max_rand_tries:
        rand_tries += 1
        v_idx = rng.randrange(n)
        if v_idx == s_idx or v_idx == z_idx:
            continue
        tt = rng.randint(1, T)
        if (v_idx, tt) in pos_set:
            continue
        key = (g_id, s_idx, z_idx, int(d), int(v_idx), int(tt))
        before = examples_store.get(key, None)
        add_labeled_example(examples_store, key, 0.0)
        if before is None and key in examples_store:
            rand_added += 1

    if rand_added < K_rand:
        log(f"    {prefix} [WARN] random negatives: added {rand_added}/{K_rand} (tries={rand_tries})")

    log(
        f"    {prefix} DATA: +{len(pos_pairs)} ILP pos, +{soft_added} soft pos, "
        f"+{hard_added} hard neg, +{rand_added} rand neg"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    validate_config()
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    max_paths_cap = _normalize_cap(ILP_MAX_PATHS)

    graphs_data: List[dict] = []
    examples_store: Dict[ExampleKey, float] = {}

    all_inputs = []
    for name, path in REAL_UVT_FILES:
        all_inputs.append(("real", name, path))
    for name, path in SYNTHETIC_UVT_FILES:
        all_inputs.append(("synthetic", name, path))

    log("=" * 80)
    log("OFFLINE JOINT REAL + SYNTHETIC DATASET BUILDER (ILP + PATH SAMPLING, CONTIGUITY)")
    log("=" * 80)
    log(f"Real graphs:      {len(REAL_UVT_FILES)}")
    log(f"Synthetic graphs: {len(SYNTHETIC_UVT_FILES)}")
    log(f"Total graph files:{len(all_inputs)}")
    log(f"OUT_DIR: {OUT_DIR}")
    log(f"Contiguity: {'ON' if ENFORCE_CONTIGUITY else 'OFF'}")
    log(f"ILP_MAX_PATHS: {ILP_MAX_PATHS} -> cap={max_paths_cap}")
    log(f"PATH_SAMPLES per query: {PATH_SAMPLES}")
    log("=" * 80)

    current_g_id = 0

    for source_type, name, uvt_file in all_inputs:
        if not os.path.exists(uvt_file):
            log(f"\n[SKIP][GRAPH {current_g_id}] missing file: {uvt_file}")
            continue

        log("\n" + "-" * 80)
        log(f"[GRAPH {current_g_id}] type={source_type} | {name} | {uvt_file}")
        log("-" * 80)

        if source_type == "real":
            edges, meta, combos = read_uvt_temporal_graph_with_combos(uvt_file)
            if not combos:
                log(f"  [SKIP][GRAPH {current_g_id}] no combos in header")
                continue

            nodes = sorted(meta["nodes_set"])
            n = len(nodes)
            T = int(meta["max_timestamp"])
            node2idx = {v: i for i, v in enumerate(nodes)}
            idx2node = {i: v for v, i in node2idx.items()}

            edges_idx = [(node2idx[u], node2idx[v], t) for (u, v, t) in edges]
            in_index, out_index = build_time_indices_mapped(edges_idx, n, T, WINDOW_W)
            tg_by_out, times_sorted = build_temporal_graph_mapped(edges_idx)

            graph_block = {
                "graph_name": name,
                "dataset_type": "real",
                "city": name,
                "uvt_file": uvt_file,
                "num_nodes": n,
                "T": T,
                "node2idx": node2idx,
                "idx2node": idx2node,
                "in_index": in_index,
                "out_index": out_index,
                "x_cache": {},
            }
            graphs_data.append(graph_block)

            rng = random.Random(SEED + 999 * current_g_id)

            for j, c in enumerate(combos, start=1):
                cid = int(c["combo_id"])
                s0, z0, d = int(c["source"]), int(c["target"]), int(c["deadline"])
                prefix = f"[GRAPH {current_g_id}][combo {cid:02d} {j}/{len(combos)}]"

                process_one_query(
                    g_id=current_g_id,
                    prefix=prefix,
                    graph_block=graph_block,
                    edges=edges,
                    edges_idx=edges_idx,
                    tg_by_out=tg_by_out,
                    times_sorted=times_sorted,
                    T=T,
                    n=n,
                    s0=s0,
                    z0=z0,
                    d=d,
                    node2idx=node2idx,
                    examples_store=examples_store,
                    max_paths_cap=max_paths_cap,
                    rng=rng,
                )

        elif source_type == "synthetic":
            edges, meta, inst = read_synthetic_uvt_single_instance(uvt_file)

            nodes = sorted(meta["nodes_set"])
            n = len(nodes)
            T = int(meta["max_timestamp"])
            node2idx = {v: i for i, v in enumerate(nodes)}
            idx2node = {i: v for v, i in node2idx.items()}

            edges_idx = [(node2idx[u], node2idx[v], t) for (u, v, t) in edges]
            in_index, out_index = build_time_indices_mapped(edges_idx, n, T, WINDOW_W)
            tg_by_out, times_sorted = build_temporal_graph_mapped(edges_idx)

            graph_block = {
                "graph_name": name,
                "dataset_type": "synthetic",
                "city": name,
                "uvt_file": uvt_file,
                "num_nodes": n,
                "T": T,
                "node2idx": node2idx,
                "idx2node": idx2node,
                "in_index": in_index,
                "out_index": out_index,
                "x_cache": {},
            }
            graphs_data.append(graph_block)

            rng = random.Random(SEED + 999 * current_g_id)

            s0 = int(inst["source"])
            z0 = int(inst["target"])
            d = int(inst["deadline"])
            prefix = f"[GRAPH {current_g_id}]"

            process_one_query(
                g_id=current_g_id,
                prefix=prefix,
                graph_block=graph_block,
                edges=edges,
                edges_idx=edges_idx,
                tg_by_out=tg_by_out,
                times_sorted=times_sorted,
                T=T,
                n=n,
                s0=s0,
                z0=z0,
                d=d,
                node2idx=node2idx,
                examples_store=examples_store,
                max_paths_cap=max_paths_cap,
                rng=rng,
            )

        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        current_g_id += 1

    # -------------------------------------------------------------------------
    # Build final tensors from examples_store (stable ordering)
    # -------------------------------------------------------------------------
    keys_sorted = sorted(examples_store.keys())
    ex_graph, ex_s, ex_z, ex_d, ex_v, ex_t, ex_y = [], [], [], [], [], [], []

    for (g_id, s_idx, z_idx, d, v_idx, tt) in keys_sorted:
        ex_graph.append(int(g_id))
        ex_s.append(int(s_idx))
        ex_z.append(int(z_idx))
        ex_d.append(int(d))
        ex_v.append(int(v_idx))
        ex_t.append(int(tt))
        ex_y.append(float(examples_store[(g_id, s_idx, z_idx, d, v_idx, tt)]))

    graphs_save = []
    for g in graphs_data:
        graphs_save.append({
            "graph_name": g["graph_name"],
            "dataset_type": g["dataset_type"],
            "city": g["city"],
            "uvt_file": g["uvt_file"],
            "num_nodes": g["num_nodes"],
            "T": g["T"],
            "node2idx": g["node2idx"],
            "idx2node": g["idx2node"],
            "in_index": g["in_index"],
            "out_index": g["out_index"],
            "x_cache": g["x_cache"],
        })

    dataset = {
        "examples": {
            "graph": torch.tensor(ex_graph, dtype=torch.long),
            "v":     torch.tensor(ex_v, dtype=torch.long),
            "t":     torch.tensor(ex_t, dtype=torch.long),
            "s":     torch.tensor(ex_s, dtype=torch.long),
            "z":     torch.tensor(ex_z, dtype=torch.long),
            "d":     torch.tensor(ex_d, dtype=torch.long),
            "y":     torch.tensor(ex_y, dtype=torch.float32),
        },
        "graphs": graphs_save,
        "meta": {
            "window_w": WINDOW_W,
            "ilp_time_limit": ILP_TIME_LIMIT,
            "ilp_max_paths": ILP_MAX_PATHS,
            "enforce_contiguity": ENFORCE_CONTIGUITY,
            "path_samples": PATH_SAMPLES,
            "sampler_tries_mult": SAMPLER_TRIES_MULT,
            "dfs_restarts_per_path": DFS_RESTARTS_PER_PATH,
            "path_max_steps": PATH_MAX_STEPS,
            "hard_neg_mult": HARD_NEG_MULT,
            "soft_pos_mult": SOFT_POS_MULT,
            "neg_ratio_random": NEG_RATIO_RANDOM,
            "seed": SEED,
            "feat_dim": 8,
            "soft_pos_min_label": SOFT_POS_MIN_LABEL,
            "soft_pos_max_label": SOFT_POS_MAX_LABEL,
            "example_key": "(g_id,s_idx,z_idx,d,v_idx,t)",
            "num_real_graphs": len(REAL_UVT_FILES),
            "num_synthetic_graphs": len(SYNTHETIC_UVT_FILES),
            "dataset_type": "joint_real_synthetic",
        }
    }

    out_pt = os.path.join(OUT_DIR, "joint_real_synthetic_dataset.pt")
    torch.save(dataset, out_pt)

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(dataset["meta"], f, indent=2)

    log("\n" + "=" * 80)
    log("[DONE] Saved joint real + synthetic dataset (ILP + path sampling, CONTIGUITY):")
    log(f"  {out_pt}")
    log(f"Total unique examples: {len(ex_y)}")
    log(f"Total graphs:         {len(graphs_save)}")
    log("=" * 80)


if __name__ == "__main__":
    main()
