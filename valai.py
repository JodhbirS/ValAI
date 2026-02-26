#!/usr/bin/env python3
# valai.py — CLI entry point (interactive solve / train).

import sys
import argparse

from constants import AGENTS, MAPS
from train import DEFAULTS



def _print_result(result: dict, map_name: str, locked: list[str]):
    team = result["team"]
    wr = result["wr"]
    sigma = result["sigma"]
    suggested = [a for a in team if a not in locked]

    print()
    print(f"  Map      : {map_name}")
    print(f"  Locked   : {', '.join(locked) if locked else '(none)'}")
    print(f"  Pick     : {', '.join(suggested)}")
    print(f"  Win Rate : {wr * 100:.2f}%")
    print(f"  SD       : {sigma:.4f}")
    print()


def _pick_map() -> str:
    sorted_maps = sorted(MAPS)
    cols, col_w = 3, 18
    rows = (len(sorted_maps) + cols - 1) // cols
    print()
    for r in range(rows):
        line = ""
        for c in range(cols):
            idx = c * rows + r
            if idx < len(sorted_maps):
                entry = f"{idx + 1:>2}. {sorted_maps[idx]}"
                line += entry.ljust(col_w)
        print(f"  {line.rstrip()}")
    print()
    while True:
        raw = input("  Map (name or number): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(sorted_maps):
                return sorted_maps[idx]
        except ValueError:
            pass
        matches = [m for m in MAPS if m.lower().startswith(raw.lower())]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  Ambiguous — did you mean: {matches}?")
            continue
        print(f"  Unknown map '{raw}'. Try again.")


def _pick_locked() -> list[str]:
    agents = list(AGENTS)
    cols, col_w = 4, 16
    rows = (len(agents) + cols - 1) // cols
    print()
    for r in range(rows):
        line = ""
        for c in range(cols):
            idx = c * rows + r
            if idx < len(agents):
                entry = f"{idx + 1:>2}. {agents[idx]}"
                line += entry.ljust(col_w)
        print(f"  {line.rstrip()}")
    print()

    while True:
        raw = input("  Lock agents (e.g. 1,5,12) or Enter for none: ").strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        locked, bad = [], []
        for p in parts:
            try:
                idx = int(p) - 1
                if 0 <= idx < len(agents):
                    locked.append(agents[idx])
                else:
                    bad.append(p)
            except ValueError:
                bad.append(p)
        if bad:
            print(f"  Invalid: {', '.join(bad)}. Pick numbers 1–{len(agents)}.")
            continue
        if len(locked) >= 5:
            print("  Max 4 agents.")
            continue
        return locked


def cmd_solve_interactive():
    import os

    if not os.path.exists("valai_model.pt"):
        print("\n  [valai] No trained model found. Run:  python valai.py train\n")
        sys.exit(1)

    print()
    print("  ValAI — Team Composition Optimizer")
    print()

    map_name = _pick_map()

    locked = _pick_locked()

    from scorer import SpinningTopScorer
    from solver import ValAISolver

    scorer = SpinningTopScorer.from_checkpoint("valai_model.pt", "agent_win_rates.csv")
    solver = ValAISolver(scorer, top_pct=0.02, max_per_role=2)
    elite = solver.solve(map_name, locked)

    if elite:
        _print_result(elite[0], map_name, locked)


def cmd_train(args):
    from train import train

    train({k: v for k, v in vars(args).items() if k != "command"})


def main():
    if len(sys.argv) == 1:
        cmd_solve_interactive()
        return

    parser = argparse.ArgumentParser(prog="valai", description="ValAI — Valorant Team Optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train the Synergy Transformer")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            tp.add_argument(f"--{k}", action="store_true", default=v)
        else:
            tp.add_argument(f"--{k}", type=type(v), default=v)

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
