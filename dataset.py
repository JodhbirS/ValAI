# dataset.py — Comp dataset with Bayesian credibility adjustment.
#
# adjusted_wr = (rounds * observed_wr + prior_n * prior_wr) / (rounds + prior_n)

import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import AGENT_TO_IDX, MAP_TO_IDX

AGENT_PRIOR_N = 800
AGENT_PRIOR_WR = 0.40
COMP_PRIOR_N = 200
COMP_PRIOR_WR = 0.50


def credibility_adjust(
    win_rate: float,
    rounds: int | float,
    prior_n: int,
    prior_wr: float,
) -> float:
    """Bayesian shrinkage toward prior_wr."""
    n = float(rounds)
    return (n * win_rate + prior_n * prior_wr) / (n + prior_n)


def _load_baseline_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["agent"] = df["agent"].str.strip()
    df["map"] = df["map"].str.strip()
    return df


def load_agent_data(
    csv_path: str = "agent_win_rates.csv",
    prior_n: int = AGENT_PRIOR_N,
    prior_wr: float = AGENT_PRIOR_WR,
) -> tuple[
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
    dict[tuple[int, int], int],
]:
    """Load agent baselines from CSV in a single pass.

    Returns:
        (adj_baselines, raw_baselines, agent_rounds)
        Keys are (map_idx, agent_idx).
    """
    df = _load_baseline_df(csv_path)
    adj: dict[tuple[int, int], float] = {}
    raw: dict[tuple[int, int], float] = {}
    rounds: dict[tuple[int, int], int] = {}
    for _, row in df.iterrows():
        if row["agent"] not in AGENT_TO_IDX or row["map"] not in MAP_TO_IDX:
            continue
        key = (MAP_TO_IDX[row["map"]], AGENT_TO_IDX[row["agent"]])
        raw_wr = float(row["win_rate"]) / 100.0
        adj[key] = credibility_adjust(raw_wr, int(row["rounds"]), prior_n, prior_wr)
        raw[key] = raw_wr
        rounds[key] = int(row["rounds"])
    return adj, raw, rounds


# Keep thin wrappers for backward compatibility (used by pair_dataset / CompDataset).
def load_agent_baselines(
    csv_path: str = "agent_win_rates.csv",
    prior_n: int = AGENT_PRIOR_N,
    prior_wr: float = AGENT_PRIOR_WR,
) -> dict[tuple[int, int], float]:
    return load_agent_data(csv_path, prior_n, prior_wr)[0]


class CompDataset(Dataset):
    """
    5-agent comp dataset. Each sample returns:
        team_ids (5,), map_id (), win_rate (), rounds (), individual_wrs (5,)

    Agent order is randomly permuted per access (free regularisation since
    the model is permutation-invariant).
    """

    def __init__(
        self,
        csv_path: str = "comp_win_rates.csv",
        baseline_csv: str = "agent_win_rates.csv",
        min_rounds: int = 20,
        agent_prior_n: int = AGENT_PRIOR_N,
        agent_prior_wr: float = AGENT_PRIOR_WR,
        comp_prior_n: int = COMP_PRIOR_N,
    ):
        self.adj_baselines = load_agent_baselines(
            baseline_csv, prior_n=agent_prior_n, prior_wr=agent_prior_wr
        )

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for col in ["map", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]:
            df[col] = df[col].str.strip()

        self.samples: list[tuple] = []
        skipped = 0
        for _, row in df.iterrows():
            agents = [row[f"agent_{i}"] for i in range(1, 6)]
            if not all(a in AGENT_TO_IDX for a in agents):
                skipped += 1
                continue
            if row["map"] not in MAP_TO_IDX:
                skipped += 1
                continue
            rounds = int(row["rounds"])
            if rounds < min_rounds:
                skipped += 1
                continue

            map_idx = MAP_TO_IDX[row["map"]]
            agent_idxs = [AGENT_TO_IDX[a] for a in agents]
            raw_wr = float(row["win_rate"]) / 100.0
            wr = credibility_adjust(raw_wr, rounds, comp_prior_n, COMP_PRIOR_WR)
            individual_wrs = [
                self.adj_baselines.get((map_idx, aid), agent_prior_wr)
                for aid in agent_idxs
            ]
            self.samples.append((agent_idxs, map_idx, wr, rounds, individual_wrs))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        agent_ids, map_id, wr, rounds, individual_wrs = self.samples[idx]

        perm = torch.randperm(5)
        agent_tensor = torch.tensor(agent_ids, dtype=torch.long)[perm]
        indwrs_tensor = torch.tensor(individual_wrs, dtype=torch.float32)[perm]

        return (
            agent_tensor,
            torch.tensor(map_id, dtype=torch.long),
            torch.tensor(wr, dtype=torch.float32),
            torch.tensor(rounds, dtype=torch.float32),
            indwrs_tensor,
        )
