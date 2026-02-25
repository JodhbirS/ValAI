# pair_dataset.py — Pairwise agent synergy data for embedding pretraining.
#
# Each 5-agent comp yields C(5,2) = 10 pairs. Aggregated by (map, agent_a, agent_b)
# using rounds-weighted mean, then credibility-adjusted.

import pandas as pd
import torch
from itertools import combinations
from collections import defaultdict
from torch.utils.data import Dataset

from constants import AGENT_TO_IDX, MAP_TO_IDX
from dataset import credibility_adjust, load_agent_baselines, AGENT_PRIOR_WR, COMP_PRIOR_N


def extract_pairs(
    csv_path: str = "comp_win_rates.csv",
    min_rounds: int = 20,
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """{(map_idx, agent_a_idx, agent_b_idx): (weighted_mean_wr, total_rounds)}"""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in ["map", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]:
        df[col] = df[col].str.strip()

    accum: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])

    for _, row in df.iterrows():
        agents = [row[f"agent_{i}"] for i in range(1, 6)]
        if not all(a in AGENT_TO_IDX for a in agents):
            continue
        if row["map"] not in MAP_TO_IDX:
            continue
        rounds = int(row["rounds"])
        if rounds < min_rounds:
            continue

        map_idx = MAP_TO_IDX[row["map"]]
        raw_wr = float(row["win_rate"]) / 100.0
        idxs = sorted([AGENT_TO_IDX[a] for a in agents])

        for a, b in combinations(idxs, 2):
            key = (map_idx, a, b)
            accum[key][0] += raw_wr * rounds
            accum[key][1] += rounds

    return {key: (wr_sum / rnd_sum, rnd_sum) for key, (wr_sum, rnd_sum) in accum.items()}


class PairDataset(Dataset):
    """Pairwise agent synergy dataset for pretraining embeddings."""

    def __init__(
        self,
        csv_path: str = "comp_win_rates.csv",
        baseline_csv: str = "agent_win_rates.csv",
        min_rounds: int = 20,
        comp_prior_n: int = COMP_PRIOR_N,
    ):
        self.adj_baselines = load_agent_baselines(baseline_csv)
        pairs = extract_pairs(csv_path, min_rounds)
        self.samples: list[tuple] = []

        for (map_idx, a, b), (raw_wr, total_rounds) in pairs.items():
            wr = credibility_adjust(raw_wr, total_rounds, comp_prior_n, 0.50)
            ind_wrs = [
                self.adj_baselines.get((map_idx, a), AGENT_PRIOR_WR),
                self.adj_baselines.get((map_idx, b), AGENT_PRIOR_WR),
            ]
            self.samples.append((a, b, map_idx, wr, total_rounds, ind_wrs))

        print(f"[pair_dataset] {len(self.samples)} unique pair-map samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        a, b, map_idx, wr, rounds, ind_wrs = self.samples[idx]
        return (
            torch.tensor(a, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(map_idx, dtype=torch.long),
            torch.tensor(wr, dtype=torch.float32),
            torch.tensor(rounds, dtype=torch.float32),
            torch.tensor(ind_wrs, dtype=torch.float32),
        )
