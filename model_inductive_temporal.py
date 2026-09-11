# model_inductive_temporal.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

Tensor = torch.Tensor


class InductiveTemporalScorer(nn.Module):


    def __init__(
        self,
        feat_dim: int,
        window_w: int,
        delta_dim: int = 16,
        msg_dim: int = 64,
        hidden_dim: int = 128,
        h_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.w = int(window_w)

        # Time-gap embedding for Δ in [0..w]
        self.delta_emb = nn.Embedding(self.w + 1, delta_dim)

        # message MLPs for incoming/outgoing arcs
        msg_inp = feat_dim + delta_dim
        self.phi_in = nn.Sequential(
            nn.Linear(msg_inp, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )
        self.phi_out = nn.Sequential(
            nn.Linear(msg_inp, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )

        # combine local summaries with center features
        comb_inp = feat_dim + msg_dim + msg_dim
        self.combine = nn.Sequential(
            nn.Linear(comb_inp, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, h_dim),
            nn.ReLU(),
        )

        # score head: [h_v(t), h_s(t), h_z(t), d_norm] -> logit
        score_inp = 3 * h_dim + 1
        self.score_head = nn.Sequential(
            nn.Linear(score_inp, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.delta_emb.weight)

    def _aggregate(
        self,
        x_tensor: Tensor,  # (N, T+1, feat_dim)
        v: int,
        t: int,
        in_arcs: List[Tuple[int, int]],   # [(u, tau)]
        out_arcs: List[Tuple[int, int]],  # [(u, tau)]
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        # incoming
        if not in_arcs:
            m_in = torch.zeros((1, self.phi_in[-1].out_features), device=device)
        else:
            u_ids = torch.tensor([u for (u, _tau) in in_arcs], dtype=torch.long, device=device)
            taus = torch.tensor([_tau for (_u, _tau) in in_arcs], dtype=torch.long, device=device)
            deltas = torch.clamp(t - taus, 0, self.w)

            x_u = x_tensor[u_ids, taus]  # (k, feat_dim)
            e_d = self.delta_emb(deltas) # (k, delta_dim)
            msg = self.phi_in(torch.cat([x_u, e_d], dim=1))
            m_in = msg.mean(dim=0, keepdim=True)

        # outgoing
        if not out_arcs:
            m_out = torch.zeros((1, self.phi_out[-1].out_features), device=device)
        else:
            u_ids = torch.tensor([u for (u, _tau) in out_arcs], dtype=torch.long, device=device)
            taus = torch.tensor([_tau for (_u, _tau) in out_arcs], dtype=torch.long, device=device)
            deltas = torch.clamp(taus - t, 0, self.w)

            x_u = x_tensor[u_ids, taus]
            e_d = self.delta_emb(deltas)
            msg = self.phi_out(torch.cat([x_u, e_d], dim=1))
            m_out = msg.mean(dim=0, keepdim=True)

        return m_in, m_out

    def compute_h(
        self,
        x_tensor: Tensor,  # (N, T+1, feat_dim)
        in_index: Dict[Tuple[int, int], List[Tuple[int, int]]],
        out_index: Dict[Tuple[int, int], List[Tuple[int, int]]],
        v: int,
        t: int,
        device: torch.device,
    ) -> Tensor:
        x_vt = x_tensor[v, t].unsqueeze(0)  # (1, feat_dim)
        in_arcs = in_index.get((v, t), [])
        out_arcs = out_index.get((v, t), [])
        m_in, m_out = self._aggregate(x_tensor, v, t, in_arcs, out_arcs, device)
        h = self.combine(torch.cat([x_vt, m_in, m_out], dim=1))
        return h

    @staticmethod
    def _deadline_norm(d_val: int, T_val: int) -> float:
        if T_val <= 0:
            return 0.0
        d_clip = min(max(int(d_val), 0), int(T_val))
        return float(d_clip) / float(T_val)

    def forward(
        self,
        batch_graph: torch.LongTensor,   # (B,)
        batch_v: torch.LongTensor,       # (B,)
        batch_t: torch.LongTensor,       # (B,)
        batch_s: torch.LongTensor,       # (B,)
        batch_z: torch.LongTensor,       # (B,)
        batch_d: torch.LongTensor,       # (B,) raw deadline integer
        graphs_data: List[dict],         # list indexed by graph_id
        device: torch.device,
    ) -> Tensor:

        B = batch_v.size(0)
        logits = []
        cache: Dict[Tuple[int, int, int], Tensor] = {}  # (g, node, t) -> h

        for i in range(B):
            g = int(batch_graph[i].item())
            v = int(batch_v[i].item())
            t = int(batch_t[i].item())
            s = int(batch_s[i].item())
            z = int(batch_z[i].item())
            d = int(batch_d[i].item())

            gd = graphs_data[g]
            x_tensor = gd["x_tensor"]
            in_index = gd["in_index"]
            out_index = gd["out_index"]
            T_val = int(gd["T"])

            def get_h(node: int, time: int) -> Tensor:
                key = (g, node, time)
                if key in cache:
                    return cache[key]
                h = self.compute_h(x_tensor, in_index, out_index, node, time, device)
                cache[key] = h
                return h

            h_vt = get_h(v, t)
            h_st = get_h(s, t)
            h_zt = get_h(z, t)

            d_norm = self._deadline_norm(d, T_val)
            d_feat = torch.tensor([[d_norm]], dtype=torch.float32, device=device)  # (1,1)

            feat = torch.cat([h_vt, h_st, h_zt, d_feat], dim=1)  # (1, 3*h_dim+1)
            logit = self.score_head(feat)  # (1,1)
            logits.append(logit)

        return torch.cat(logits, dim=0).squeeze(1)  # (B,)
