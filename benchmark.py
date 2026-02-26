#!/usr/bin/env python3
# benchmark.py — Compare ValAI Transformer vs naive baselines on comp_win_rates.csv.

import torch
import pandas as pd

from constants import AGENT_TO_IDX, MAP_TO_IDX
from dataset import load_agent_data, credibility_adjust, COMP_PRIOR_N, COMP_PRIOR_WR
from scorer import SpinningTopScorer


def main():
    # Load actual comp data
    df = pd.read_csv("comp_win_rates.csv")
    df.columns = df.columns.str.strip()
    for col in ["map", "agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]:
        df[col] = df[col].str.strip()

    adj_baselines, raw_baselines, _ = load_agent_data("agent_win_rates.csv")

    # Load trained model
    scorer = SpinningTopScorer.from_checkpoint("valai_model.pt", "agent_win_rates.csv")

    naive_sq, avg_sq, model_sq = 0.0, 0.0, 0.0
    naive_abs, avg_abs, model_abs = 0.0, 0.0, 0.0
    n = 0

    for _, row in df.iterrows():
        agents = [row[f"agent_{i}"] for i in range(1, 6)]
        map_name = row["map"]
        if not all(a in AGENT_TO_IDX for a in agents) or map_name not in MAP_TO_IDX:
            continue

        rounds = int(row["rounds"])
        raw_wr = float(row["win_rate"]) / 100.0
        actual = credibility_adjust(raw_wr, rounds, COMP_PRIOR_N, COMP_PRIOR_WR)

        map_idx = MAP_TO_IDX[map_name]
        agent_ids = [AGENT_TO_IDX[a] for a in agents]

        # Naive: just predict 50% for everything
        naive_pred = 0.50

        # Average solo WR baseline
        solo_wrs = [raw_baselines.get((map_idx, aid), 0.50) for aid in agent_ids]
        avg_pred = sum(solo_wrs) / 5.0

        # ValAI model
        team_tensor = torch.tensor(agent_ids, dtype=torch.long)
        model_pred, _ = scorer.score(team_tensor, map_idx)

        naive_sq += (naive_pred - actual) ** 2
        avg_sq += (avg_pred - actual) ** 2
        model_sq += (model_pred - actual) ** 2

        naive_abs += abs(naive_pred - actual)
        avg_abs += abs(avg_pred - actual)
        model_abs += abs(model_pred - actual)

        n += 1

    naive_rmse = (naive_sq / n) ** 0.5 * 100
    avg_rmse = (avg_sq / n) ** 0.5 * 100
    model_rmse = (model_sq / n) ** 0.5 * 100

    naive_mae = naive_abs / n * 100
    avg_mae = avg_abs / n * 100
    model_mae = model_abs / n * 100

    print(f"\n  Benchmark on {n} compositions")
    print(f"  {'Method':<22} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 40)
    print(f"  {'Naive (always 50%)':<22} {naive_rmse:>7.2f}% {naive_mae:>7.2f}%")
    print(f"  {'Avg Solo WR':<22} {avg_rmse:>7.2f}% {avg_mae:>7.2f}%")
    print(f"  {'ValAI Transformer':<22} {model_rmse:>7.2f}% {model_mae:>7.2f}%")
    print()

    improvement = avg_rmse - model_rmse
    pct_better = (1 - model_rmse / avg_rmse) * 100
    print(f"  ValAI vs Avg Solo WR:  -{improvement:.2f}% RMSE  ({pct_better:.1f}% better)")

    improvement_naive = naive_rmse - model_rmse
    pct_better_naive = (1 - model_rmse / naive_rmse) * 100
    print(f"  ValAI vs Naive 50%:    -{improvement_naive:.2f}% RMSE  ({pct_better_naive:.1f}% better)")
    print()


if __name__ == "__main__":
    main()
