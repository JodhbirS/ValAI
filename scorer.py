# scorer.py — Batched team scorer.

import torch

from model import ValAITransformer
from dataset import load_agent_data
from constants import NUM_AGENTS, NUM_MAPS


class SpinningTopScorer:
    """
    Scores teams by predicted win rate and solo-WR spread (sigma).

    Args:
        model:          Eval-mode ValAITransformer
        adj_baselines:  {(map_idx, agent_idx): credibility_adjusted_wr}
        raw_baselines:  {(map_idx, agent_idx): raw_observed_wr}
        agent_rounds:   {(map_idx, agent_idx): rounds_played}
    """

    def __init__(self, model, adj_baselines, raw_baselines, agent_rounds):
        self.model = model
        self.device = next(model.parameters()).device
        self.adj_baselines = adj_baselines
        self.raw_baselines = raw_baselines
        self.agent_rounds = agent_rounds

        self.adj_matrix = torch.zeros(NUM_AGENTS, NUM_MAPS)
        self.raw_matrix = torch.zeros(NUM_AGENTS, NUM_MAPS)
        for (map_idx, agent_idx), wr in adj_baselines.items():
            self.adj_matrix[agent_idx, map_idx] = wr
        for (map_idx, agent_idx), wr in raw_baselines.items():
            self.raw_matrix[agent_idx, map_idx] = wr

    @torch.no_grad()
    def score(self, team_ids: torch.Tensor, map_idx: int) -> tuple[float, float]:
        """Score a single team. Returns (predicted_wr, sigma)."""
        wrs, sigmas = self.score_batch(team_ids.unsqueeze(0), map_idx)
        return wrs[0].item(), sigmas[0].item()

    @torch.no_grad()
    def score_batch(
        self,
        team_ids_batch: torch.Tensor,
        map_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score N teams. Returns (wr [N], sigma [N]) on CPU."""
        self.model.eval()
        N = team_ids_batch.shape[0]

        adj_wrs = self.adj_matrix[team_ids_batch, map_idx]

        t_in = team_ids_batch.to(self.device)
        m_in = torch.full((N,), map_idx, dtype=torch.long, device=self.device)
        v_in = adj_wrs.to(self.device)

        wr_team = self.model(t_in, m_in, v_in).clamp(0.0, 1.0)

        raw_wrs = self.raw_matrix[team_ids_batch, map_idx].to(self.device)
        sigma = raw_wrs.std(dim=1, correction=0)

        return wr_team.cpu(), sigma.cpu()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = "valai_model.pt",
        baseline_csv: str = "agent_win_rates.csv",
    ) -> "SpinningTopScorer":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {})
        model = ValAITransformer(
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 8),
            layers=cfg.get("layers", 4),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        adj, raw, rounds = load_agent_data(baseline_csv)
        return cls(
            model=model,
            adj_baselines=adj,
            raw_baselines=raw,
            agent_rounds=rounds,
        )
