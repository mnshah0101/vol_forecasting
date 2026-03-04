"""
MoE-GRU: Mixture of Experts with Horizon-Routed GRU Experts
============================================================
Different prediction horizons (1s vs 15m) require fundamentally different
temporal dynamics. This model routes each sample to specialized GRU experts
based on the horizon embedding, using top-k gating with load balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUExpert(nn.Module):
    """A single GRU expert: GRU encoder + MLP head -> scalar."""

    def __init__(self, d_proj: int, hidden_size: int, num_layers: int,
                 dropout: float, head_units: list[int]):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_proj,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d = hidden_size
        layers = []
        for units in head_units:
            layers.extend([nn.Linear(d, units), nn.ReLU(), nn.Dropout(dropout)])
            d = units
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """x: (B_subset, L, d_proj) -> (B_subset, 1) or ((B_subset, 1), (B_subset, H))"""
        _, h_n = self.gru(x)
        h = h_n[-1]  # (B_subset, hidden_size)
        pred = self.head(h)  # (B_subset, 1)
        if return_embedding:
            return pred, h
        return pred


class MoEGRU(nn.Module):
    def __init__(
        self,
        lookback: int = 1800,
        n_features: int = 50,
        n_experts: int = 4,
        top_k: int = 2,
        hidden_size: int = 32,
        d_proj: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_units: list[int] = [32],
        max_horizon: int = 900,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss = None  # set during forward when training
        self._routing_log = []  # list of (horizons, top_k_indices, top_k_weights) tensors

        # Shared input projection
        self.input_proj = nn.Linear(n_features, d_proj)

        # Horizon embedding
        self.horizon_embed = nn.Embedding(max_horizon + 1, d_proj)

        # Gating network: horizon embedding -> expert logits
        self.gate = nn.Linear(d_proj, n_experts)

        # Expert GRUs
        self.experts = nn.ModuleList([
            GRUExpert(d_proj, hidden_size, num_layers, dropout, list(head_units))
            for _ in range(n_experts)
        ])

        param_count = sum(p.numel() for p in self.parameters())
        print(f"MoEGRU: {n_experts} experts (top-{top_k}), "
              f"d_proj={d_proj}, hidden={hidden_size}, layers={num_layers}, "
              f"head={head_units}, aux_weight={aux_loss_weight}, "
              f"params={param_count:,}")

    @property
    def embedding_dim(self) -> int:
        return self.experts[0].gru.hidden_size

    def forward(self, x: torch.Tensor, horizon=None, return_embedding: bool = False):
        # Run entirely in fp32 (same rationale as GRU — sequential scan)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            B = x.shape[0]

            # 1. Shared input projection
            x_proj = self.input_proj(x)  # (B, L, d_proj)

            # 2. Horizon embedding + additive injection
            if horizon is not None:
                h_embed = self.horizon_embed(horizon)  # (B, d_proj)
                x_proj = x_proj + h_embed.unsqueeze(1)  # broadcast over L
            else:
                h_embed = self.input_proj.weight.new_zeros(B, self.input_proj.out_features)

            # 3. Gating: horizon embedding -> top-k expert weights
            gate_logits = self.gate(h_embed)  # (B, n_experts)
            top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)  # (B, k)
            top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B, k)

            # 4. Load balancing auxiliary loss (Switch Transformer style)
            if self.training:
                self.aux_loss = self._load_balance_loss(gate_logits, top_k_indices)
            else:
                self.aux_loss = None

            # 4b. Store routing decisions for diagnostics
            if horizon is not None and len(self._routing_log) < 500:
                self._routing_log.append((
                    horizon.detach().cpu(),
                    top_k_indices.detach().cpu(),
                    top_k_weights.detach().cpu(),
                ))

            # 5. Dispatch to experts and combine
            output = x_proj.new_zeros(B)  # (B,)
            emb_dim = self.embedding_dim
            embedding = x_proj.new_zeros(B, emb_dim) if return_embedding else None

            for e_idx in range(self.n_experts):
                # Build mask: which (sample, slot) pairs route to this expert
                # top_k_indices: (B, k), check if any slot == e_idx
                expert_mask = (top_k_indices == e_idx)  # (B, k) bool
                sample_uses_expert = expert_mask.any(dim=-1)  # (B,) bool

                if not sample_uses_expert.any():
                    continue

                # Gather the samples assigned to this expert
                sample_indices = sample_uses_expert.nonzero(as_tuple=True)[0]  # (n_assigned,)
                expert_input = x_proj[sample_indices]  # (n_assigned, L, d_proj)

                # Run expert
                expert_result = self.experts[e_idx](expert_input, return_embedding=return_embedding)
                if return_embedding:
                    expert_out, expert_emb = expert_result
                    expert_out = expert_out.squeeze(-1)  # (n_assigned,)
                else:
                    expert_out = expert_result.squeeze(-1)  # (n_assigned,)

                # Get the corresponding gate weight for this expert
                # expert_mask[sample_indices] tells us which slot(s) matched
                slot_weights = top_k_weights[sample_indices]  # (n_assigned, k)
                slot_mask = expert_mask[sample_indices]  # (n_assigned, k) bool
                # Sum weights across slots where this expert was selected
                # (handles rare case where same expert appears in multiple top-k slots)
                weights = (slot_weights * slot_mask.float()).sum(dim=-1)  # (n_assigned,)

                # Scatter weighted output back
                output.scatter_add_(0, sample_indices, weights * expert_out)

                # Scatter weighted embeddings back
                if return_embedding:
                    weighted_emb = expert_emb * weights.unsqueeze(1)  # (n_assigned, H)
                    embedding.scatter_add_(0, sample_indices.unsqueeze(1).expand_as(weighted_emb), weighted_emb)

            if return_embedding:
                return output, embedding
            return output

    def _load_balance_loss(self, gate_logits: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """Switch Transformer style load balancing loss.

        Encourages uniform expert utilization by penalizing the product of:
        - f_i: fraction of tokens routed to expert i
        - P_i: mean gate probability for expert i
        """
        B = gate_logits.shape[0]
        N = self.n_experts

        # f_i: fraction of tokens dispatched to each expert
        # Count how many samples have expert i in their top-k
        one_hot = F.one_hot(top_k_indices, N).float()  # (B, k, N)
        tokens_per_expert = one_hot.sum(dim=(0, 1))  # (N,)
        f = tokens_per_expert / (B * self.top_k)  # (N,)

        # P_i: mean routing probability for each expert (over full softmax)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, N)
        P = gate_probs.mean(dim=0)  # (N,)

        return self.aux_loss_weight * N * (f * P).sum()

    def routing_summary(self, horizon_buckets=None, clear=True) -> str:
        """Summarize routing decisions as a horizon→expert table.

        Args:
            horizon_buckets: list of (label, min_sec, max_sec) tuples.
                Defaults to [1s, 5s, 30s, 1m, 5m, 15m] buckets.
            clear: whether to clear the routing log after summarizing.

        Returns:
            Formatted string table.
        """
        if not self._routing_log:
            return "  (no routing data collected)"

        all_h = torch.cat([r[0] for r in self._routing_log])          # (N_total,)
        all_idx = torch.cat([r[1] for r in self._routing_log])        # (N_total, k)
        all_w = torch.cat([r[2] for r in self._routing_log])          # (N_total, k)

        if clear:
            self._routing_log.clear()

        if horizon_buckets is None:
            horizon_buckets = [
                ("1s",    1,   1),
                ("2-5s",  2,   5),
                ("6-30s", 6,  30),
                ("31s-1m",31, 60),
                ("1-5m",  61, 300),
                ("5-15m", 301,900),
            ]

        N = self.n_experts
        lines = []

        # Header
        expert_hdr = "  ".join(f"E{i:d}" for i in range(N))
        lines.append(f"  {'Horizon':<10s}  {'Count':>6s}  {expert_hdr}  | top-1")

        for label, lo, hi in horizon_buckets:
            mask = (all_h >= lo) & (all_h <= hi)
            cnt = mask.sum().item()
            if cnt == 0:
                continue

            # Weighted expert utilization: sum gate weights per expert
            sub_idx = all_idx[mask]   # (cnt, k)
            sub_w = all_w[mask]       # (cnt, k)
            expert_weight = torch.zeros(N)
            for e in range(N):
                expert_weight[e] = sub_w[sub_idx == e].sum().item()
            total = expert_weight.sum().item()
            if total > 0:
                expert_pct = expert_weight / total * 100

            # Top-1 assignment (which expert gets highest weight per sample)
            top1 = sub_idx[:, 0]  # first slot = highest weight
            top1_counts = torch.zeros(N)
            for e in range(N):
                top1_counts[e] = (top1 == e).sum().item()
            top1_pct = top1_counts / cnt * 100

            pct_str = "  ".join(f"{p:5.1f}%" for p in expert_pct)
            top1_str = "  ".join(f"{p:4.0f}%" for p in top1_pct)
            lines.append(f"  {label:<10s}  {cnt:6d}  {pct_str}  | {top1_str}")

        # Overall
        total_n = len(all_h)
        expert_total = torch.zeros(N)
        for e in range(N):
            expert_total[e] = all_w[all_idx == e].sum().item()
        total_sum = expert_total.sum().item()
        if total_sum > 0:
            expert_total_pct = expert_total / total_sum * 100
        pct_str = "  ".join(f"{p:5.1f}%" for p in expert_total_pct)
        lines.append(f"  {'TOTAL':<10s}  {total_n:6d}  {pct_str}")

        return "\n".join(lines)
