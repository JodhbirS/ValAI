# model.py — Permutation-invariant Synergy Transformer.
#
# Token sequence: [MAP | agent_0 | agent_1 | agent_2 | agent_3 | agent_4]
# Each agent token = agent_emb + role_emb + baseline_proj(solo_wr)
# Map token = map_emb(map_id)
#
# Self-attention over all 6 tokens, mean-pool agent outputs only → predicted WR.

import torch
import torch.nn as nn

from constants import NUM_AGENTS, NUM_MAPS, NUM_ROLES, AGENT_ROLES, AGENT_TO_IDX


class ValAITransformer(nn.Module):
    """
    Predicts team win rate from 5 agent IDs, a map ID, and per-agent solo WRs.

    Inputs:
        team_ids        (B, 5)  int
        map_id          (B,)    int
        individual_wrs  (B, 5)  float

    Output: (B,) float — predicted win rate (unclamped during training)
    """

    def __init__(
        self,
        num_agents: int = NUM_AGENTS,
        num_maps: int = NUM_MAPS,
        num_roles: int = NUM_ROLES,
        d_model: int = 128,
        nhead: int = 8,
        layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.agent_emb = nn.Embedding(num_agents, d_model)
        self.role_emb = nn.Embedding(num_roles, d_model)
        self.baseline_proj = nn.Linear(1, d_model, bias=False)
        self.map_emb = nn.Embedding(num_maps, d_model)

        # Non-trainable buffer: agent_idx → role_idx
        role_lookup = torch.zeros(num_agents, dtype=torch.long)
        for name, role_idx in AGENT_ROLES.items():
            if name in AGENT_TO_IDX:
                role_lookup[AGENT_TO_IDX[name]] = role_idx
        self.register_buffer("role_lookup", role_lookup)

        self.embed_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.agent_emb.weight, std=0.02)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        nn.init.normal_(self.map_emb.weight, std=0.02)
        nn.init.normal_(self.baseline_proj.weight, std=0.02)

    def forward(
        self,
        team_ids: torch.Tensor,
        map_id: torch.Tensor,
        individual_wrs: torch.Tensor,
    ) -> torch.Tensor:
        # Agent tokens: embedding + role + solo WR projection
        a = self.agent_emb(team_ids)                          # (B, 5, d)
        r = self.role_emb(self.role_lookup[team_ids])         # (B, 5, d)
        v = self.baseline_proj(individual_wrs.unsqueeze(-1))  # (B, 5, d)
        agent_tokens = a + r + v

        # Map as a 6th token
        map_token = self.map_emb(map_id).unsqueeze(1)  # (B, 1, d)

        # Attention over [map, agent_0..4], pool agents only
        x = torch.cat([map_token, agent_tokens], dim=1)  # (B, 6, d)
        x = self.embed_norm(x)
        x = self.transformer(x)                           # (B, 6, d)
        team_vec = x[:, 1:].mean(dim=1)                   # (B, d)

        return self.output_head(team_vec).squeeze(-1)  # (B,)
