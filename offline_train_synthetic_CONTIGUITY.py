

import os
import sys
import json
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# SETUP (EDIT HERE ONLY)
# -----------------------------
DATA_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/ML_result/joint_real_synthetic_dataset_contiguity"
OUT_DIR  = "/Users/mehdi/Desktop/Network_Analysis/TSD/ML_result/trained_model_joint_real_synthetic_contiguity"

MODEL_CODE_DIR = "/Users/mehdi/Desktop/Network_Analysis/TSD/python_v2"
sys.path.append(MODEL_CODE_DIR)
from model_inductive_temporal import InductiveTemporalScorer  # noqa

EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model sizes
DELTA_DIM = 16
MSG_DIM = 64
HIDDEN_DIM = 128
H_DIM = 64
DROPOUT = 0.1

# optional: cache (graph,s,z) tensors on GPU to avoid repeated .to(device)
MAX_GPU_CACHE_ITEMS = 32
# -----------------------------


class TinyLRUDeviceCache:
    """Very small LRU cache for GPU tensors keyed by (gi,si,zi)."""
    def __init__(self, max_items: int):
        self.max_items = int(max_items)
        self._d: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self._order: List[Tuple[int, int, int]] = []

    def get(self, key):
        if key not in self._d:
            return None
        self._order.remove(key)
        self._order.append(key)
        return self._d[key]

    def put(self, key, value: torch.Tensor):
        if self.max_items <= 0:
            return
        if key in self._d:
            self._order.remove(key)
        self._d[key] = value
        self._order.append(key)
        while len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._d.pop(old, None)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    ds = torch.load(os.path.join(DATA_DIR, "joint_real_synthetic_dataset.pt"), map_location="cpu")
    ex = ds["examples"]
    graphs = ds["graphs"]
    meta = ds.get("meta", {})

    feat_dim = int(meta.get("feat_dim", 8))
    w = int(meta.get("window_w", 5))

    # Runtime graph objects
    graphs_data = []
    for g in graphs:
        graphs_data.append({
            "graph_name": g.get("graph_name", g.get("city", "")),
            "dataset_type": g.get("dataset_type", "unknown"),
            "city": g.get("city", ""),
            "uvt_file": g.get("uvt_file", ""),
            "num_nodes": g["num_nodes"],
            "T": int(g["T"]),
            "node2idx": g["node2idx"],
            "idx2node": g["idx2node"],
            "in_index": g["in_index"],
            "out_index": g["out_index"],
            "x_cache": g["x_cache"],   # dict key "s_z" -> Tensor (CPU)
            "x_tensor": None,          # will be set per-group
        })

    # Build dataset tensors
    g_all = ex["graph"].long()
    v_all = ex["v"].long()
    t_all = ex["t"].long()
    s_all = ex["s"].long()
    z_all = ex["z"].long()
    d_all = ex["d"].long()
    y_all = ex["y"].float()

    N = y_all.numel()

    # Small reporting only
    num_real_graphs = sum(1 for g in graphs_data if g.get("dataset_type") == "real")
    num_synth_graphs = sum(1 for g in graphs_data if g.get("dataset_type") == "synthetic")

    loader = DataLoader(
        TensorDataset(g_all, v_all, t_all, s_all, z_all, d_all, y_all),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = InductiveTemporalScorer(
        feat_dim=feat_dim,
        window_w=w,
        delta_dim=DELTA_DIM,
        msg_dim=MSG_DIM,
        hidden_dim=HIDDEN_DIM,
        h_dim=H_DIM,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    gpu_cache = TinyLRUDeviceCache(MAX_GPU_CACHE_ITEMS)

    print("=" * 80)
    print("TRAINING (INDUCTIVE JOINT REAL + SYNTHETIC, GENERAL DEADLINE, CONTIGUITY)")
    print("=" * 80)
    print(f"Examples: {N}")
    print(f"Graphs: {len(graphs_data)} | real={num_real_graphs} | synthetic={num_synth_graphs}")
    print(f"feat_dim: {feat_dim} | window_w: {w}")
    print(f"Device: {DEVICE}")
    print(f"MAX_GPU_CACHE_ITEMS: {MAX_GPU_CACHE_ITEMS}")
    print("=" * 80)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        seen = 0

        for (g_id, v, t, s, z, d_raw, yb) in loader:
            # keep CPU copies for grouping keys
            g_id_cpu = g_id
            s_cpu = s
            z_cpu = z

            # move batch tensors to device
            g_id = g_id.to(device)
            v = v.to(device)
            t = t.to(device)
            s = s.to(device)
            z = z.to(device)
            d_raw = d_raw.to(device)
            yb = yb.to(device)

            # group indices by (graph_id, s, z)
            groups: Dict[Tuple[int, int, int], List[int]] = {}
            B = int(g_id_cpu.size(0))
            for i in range(B):
                gi = int(g_id_cpu[i].item())
                si = int(s_cpu[i].item())
                zi = int(z_cpu[i].item())
                key = (gi, si, zi)
                groups.setdefault(key, []).append(i)

            optim.zero_grad()

            logits_parts = []
            y_parts = []

            for (gi, si, zi), idxs in groups.items():
                cache_key = f"{si}_{zi}"
                if cache_key not in graphs_data[gi]["x_cache"]:
                    raise KeyError(
                        f"Missing x_cache for graph={gi} key={cache_key}. "
                        f"Builder did not store features for this (s,z)."
                    )

                # get x tensor on device (with tiny LRU)
                dev_key = (gi, si, zi)
                x_dev = gpu_cache.get(dev_key)
                if x_dev is None:
                    x_cpu = graphs_data[gi]["x_cache"][cache_key]  # CPU tensor
                    x_dev = x_cpu.to(device)
                    gpu_cache.put(dev_key, x_dev)

                # set current graph tensor for this group
                graphs_data[gi]["x_tensor"] = x_dev

                sel = torch.tensor(idxs, dtype=torch.long, device=device)

                logits_g = model(
                    batch_graph=g_id.index_select(0, sel),
                    batch_v=v.index_select(0, sel),
                    batch_t=t.index_select(0, sel),
                    batch_s=s.index_select(0, sel),
                    batch_z=z.index_select(0, sel),
                    batch_d=d_raw.index_select(0, sel),
                    graphs_data=graphs_data,
                    device=device,
                )

                logits_parts.append(logits_g)
                y_parts.append(yb.index_select(0, sel))

            logits_all = torch.cat(logits_parts, dim=0)
            y_all_b = torch.cat(y_parts, dim=0)

            loss = criterion(logits_all, y_all_b)
            loss.backward()
            optim.step()

            total += float(loss.item()) * B
            seen += B

        print(f"Epoch {epoch:03d}/{EPOCHS}  loss={total / max(1, seen):.6f}")

    save_path = os.path.join(OUT_DIR, "model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {
            "window_w": w,
            "feat_dim": feat_dim,
            "contiguity": True,
            "x_cache_mode": "in_pt",
            "dataset_type": "joint_real_synthetic",
        },
        "hparams": {
            "delta_dim": DELTA_DIM,
            "msg_dim": MSG_DIM,
            "hidden_dim": HIDDEN_DIM,
            "h_dim": H_DIM,
            "dropout": DROPOUT,
        }
    }, save_path)

    with open(os.path.join(OUT_DIR, "train_meta.json"), "w") as f:
        json.dump({
            "DATA_DIR": DATA_DIR,
            "OUT_DIR": OUT_DIR,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LR": LR,
            "DEVICE": DEVICE,
            "MAX_GPU_CACHE_ITEMS": MAX_GPU_CACHE_ITEMS,
            "dataset_meta": meta,
            "num_examples": int(N),
            "num_graphs": int(len(graphs_data)),
            "num_real_graphs": int(num_real_graphs),
            "num_synthetic_graphs": int(num_synth_graphs),
        }, f, indent=2)

    print("\n[DONE] Saved inductive joint real + synthetic model (general deadline, contiguity):")
    print(f"  {save_path}")


if __name__ == "__main__":
    main()
