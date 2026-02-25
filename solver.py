# solver.py — Enumerates valid team completions and ranks by predicted WR.

import itertools
from collections import Counter

import torch
from tqdm import tqdm

from constants import AGENTS, AGENT_TO_IDX, IDX_TO_AGENT, MAP_TO_IDX, AGENT_ROLES, ROLE_NAMES
from scorer import SpinningTopScorer

_BATCH_SIZE = 512
_CONTROLLER, _DUELIST = 0, 1


class ValAISolver:

    def __init__(
        self,
        scorer: SpinningTopScorer,
        top_pct: float = 0.02,
        max_per_role: int = 2,
    ):
        self.scorer = scorer
        self.top_pct = top_pct
        self.max_per_role = max_per_role
        self._all_ids = list(range(len(AGENTS)))

        self._role_of: dict[int, int] = {}
        for name, role_idx in AGENT_ROLES.items():
            if name in AGENT_TO_IDX:
                self._role_of[AGENT_TO_IDX[name]] = role_idx

    def _viable_agents(self, map_idx: int) -> list[int]:
        """Agents with at least 1 round on this map."""
        return [
            aid for aid in self._all_ids
            if self.scorer.agent_rounds.get((map_idx, aid), 0) > 0
        ]

    def _valid_role_comp(self, team_ids: list[int]) -> bool:
        """Max N per role, at least 1 controller and 1 duelist."""
        role_counts = Counter(self._role_of.get(aid, -1) for aid in team_ids)
        if any(c > self.max_per_role for c in role_counts.values()):
            return False
        return role_counts.get(_CONTROLLER, 0) >= 1 and role_counts.get(_DUELIST, 0) >= 1

    def solve(self, map_name: str, locked_names: list[str]) -> list[dict]:
        """Returns top compositions ranked by predicted WR."""
        if map_name not in MAP_TO_IDX:
            raise ValueError(f"Unknown map '{map_name}'. Choose from: {sorted(MAP_TO_IDX)}")

        locked_ids = []
        for name in locked_names:
            if name not in AGENT_TO_IDX:
                raise ValueError(f"Unknown agent '{name}'. Check spelling.")
            locked_ids.append(AGENT_TO_IDX[name])

        map_idx = MAP_TO_IDX[map_name]
        slots = 5 - len(locked_ids)

        viable = self._viable_agents(map_idx)
        available = [aid for aid in viable if aid not in locked_ids]
        excluded = len(self._all_ids) - len(viable)

        # Filter by role composition
        valid_combos = []
        role_rejected = 0
        for combo in itertools.combinations(available, slots):
            if self._valid_role_comp(locked_ids + list(combo)):
                valid_combos.append(combo)
            else:
                role_rejected += 1

        N = len(valid_combos)
        print(
            f"  Viable agents on {map_name}: {len(viable)}/28 "
            f"({excluded} excluded — zero picks on this map)"
        )
        print(
            f"  Role filter: {role_rejected:,} invalid comps removed "
            f"(max {self.max_per_role} per role, need ≥1 smoke + ≥1 duelist)"
        )
        print(f"  Analysing {N:,} valid completions...")

        # Build team tensor and batch-score
        all_teams = torch.zeros(N, 5, dtype=torch.long)
        for i, combo in enumerate(valid_combos):
            all_teams[i] = torch.tensor(locked_ids + list(combo), dtype=torch.long)

        all_wrs = torch.zeros(N)
        all_sigmas = torch.zeros(N)
        for start in tqdm(range(0, N, _BATCH_SIZE), desc="Scoring", ncols=70, leave=False):
            end = min(start + _BATCH_SIZE, N)
            wrs, sigmas = self.scorer.score_batch(all_teams[start:end], map_idx)
            all_wrs[start:end] = wrs
            all_sigmas[start:end] = sigmas

        # Top results
        order = all_wrs.argsort(descending=True)
        elite_n = max(1, int(N * self.top_pct))

        results = []
        for i in order[:elite_n].tolist():
            team_ids = all_teams[i].tolist()
            wr = all_wrs[i].item()
            sigma = all_sigmas[i].item()

            lifts = {}
            roles = {}
            for aid in team_ids:
                agent_name = IDX_TO_AGENT[aid]
                raw_base = self.scorer.raw_baselines.get((map_idx, aid), 0.0)
                lifts[agent_name] = round((wr - raw_base) * 100, 2)
                roles[agent_name] = ROLE_NAMES.get(self._role_of.get(aid, -1), "Unknown")

            results.append({
                "team": [IDX_TO_AGENT[i] for i in team_ids],
                "wr": wr,
                "sigma": sigma,
                "lifts": lifts,
                "roles": roles,
            })

        return results
