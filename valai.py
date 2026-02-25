#!/usr/bin/env python3
# valai.py — CLI entry point (interactive solve / train).

import sys
import argparse

from constants import AGENTS, MAPS
from train import DEFAULTS

_ROLE_TAG = {"Controller": "SMK", "Duelist": "DUE", "Initiator": "INI", "Sentinel": "SEN"}


def _bar(value: float, width: int = 20, lo: float = 0.0, hi: float = 1.0) -> str:
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def _print_result(result: dict, map_name: str, locked: list[str], rank: int = 1):
    team = result["team"]
    wr = result["wr"]
    sigma = result["sigma"]
    lifts = result["lifts"]
    roles = result.get("roles", {})
    suggested = [a for a in team if a not in locked]

    print()
    print("═" * 62)
    print(f"  ValAI · {map_name}   Rank #{rank}")
    print("═" * 62)
    tagged = [f"{a} [{_ROLE_TAG.get(roles.get(a, ''), '???')}]" for a in team]
    print(f"  {'  |  '.join(tagged)}")
    print("─" * 62)
    print(f"  Locked : {', '.join(locked) if locked else '(none)'}")
    print(f"  Pick   : {', '.join(suggested)}")
    print()
    print(f"  Win Rate  : {wr * 100:5.2f}%   {_bar(wr, lo=0.45, hi=0.58)}")
    print(f"  Spread (σ): {sigma:.4f}     {'↓ Even' if sigma < 0.03 else '↑ Uneven'}")
    print()
    print("  SYNERGY BREAKDOWN  (lift = Team WR − agent solo WR)")
    print(f"  {'Agent':<12} {'Role':<6} {'Solo WR':>8}  {'Team WR':>8}  {'Lift':>7}")
    print("  " + "─" * 52)
    for agent in team:
        raw_base = wr - (lifts[agent] / 100)
        lift_val = lifts[agent]
        sign = "+" if lift_val >= 0 else ""
        role_tag = _ROLE_TAG.get(roles.get(agent, ""), "???")
        note = "  ← below solo" if lift_val < -1.0 else ""
        print(
            f"  {agent:<12} {role_tag:<6} {raw_base * 100:>7.2f}%  {wr * 100:>7.2f}%  "
            f"{sign}{lift_val:>5.2f}%{note}"
        )
    print("═" * 62)


def _pick_map() -> str:
    """Prompt user to choose a map."""
    sorted_maps = sorted(MAPS)
    print("\n  Available maps:")
    for i, m in enumerate(sorted_maps, 1):
        print(f"    {i:>2}. {m}")
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
    """Prompt user for 0–4 locked agents."""
    print("\n  Lock agents (comma-separated), or press Enter for none:")
    print(f"  Agents: {', '.join(sorted(AGENTS))}\n")
    while True:
        raw = input("  Locked agents: ").strip()
        if not raw:
            return []
        names = [n.strip() for n in raw.split(",") if n.strip()]
        lookup = {a.lower(): a for a in AGENTS}
        fixed, bad = [], []
        for n in names:
            if n.lower() in lookup:
                fixed.append(lookup[n.lower()])
            else:
                bad.append(n)
        if bad:
            print(f"  Unknown agent(s): {bad}. Check spelling.")
            continue
        if len(fixed) >= 5:
            print("  Can lock at most 4 agents (need at least 1 open slot).")
            continue
        return fixed


def cmd_solve_interactive():
    import os

    if not os.path.exists("valai_model.pt"):
        print("\n  [valai] No trained model found. Run:  python valai.py train\n")
        sys.exit(1)

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║   ValAI — Team Composition Optimizer  ║")
    print("  ╚══════════════════════════════════════╝")

    map_name = _pick_map()
    locked = _pick_locked()

    print(f"\n  Map    : {map_name}")
    print(f"  Locked : {', '.join(locked) if locked else '(none — full 5-man search)'}\n")

    from scorer import SpinningTopScorer
    from solver import ValAISolver

    scorer = SpinningTopScorer.from_checkpoint("valai_model.pt", "agent_win_rates.csv")
    solver = ValAISolver(scorer, top_pct=0.02, max_per_role=2)
    elite = solver.solve(map_name, locked)

    show = min(3, len(elite))
    for i, result in enumerate(elite[:show], 1):
        _print_result(result, map_name, locked, rank=i)

    print(f"\n  (Top {show} of {len(elite)} valid compositions)\n")


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
