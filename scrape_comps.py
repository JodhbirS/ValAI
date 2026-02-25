# scrape_comps.py — Scrape team composition win rates from rib.gg → comp_win_rates.csv

import requests
import pandas as pd
from itertools import product

from constants import API_AGENT_NAMES, API_MAP_IDS, API_REGIONS

HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.rib.gg",
    "Referer": "https://www.rib.gg/analytics/comps",
}

THRESHOLDS = [1, 30]
SIDES = [None, "atk", "def"]  # None first → overall is canonical
REGIONS = [None] + list(API_REGIONS.keys())


def main():
    seen: dict[tuple, dict] = {}

    for map_api_id, map_name in API_MAP_IDS.items():
        map_seen: set[tuple] = set()
        query_count = 0

        for threshold, side, region in product(THRESHOLDS, SIDES, REGIONS):
            params: dict = {"mapId": map_api_id, "threshold": threshold}
            if side:
                params["side"] = side
            if region:
                params["regionId"] = region

            try:
                resp = requests.get(
                    "https://be-prod.rib.gg/v1/x/composition-win-rates",
                    params=params, headers=HEADERS, timeout=12,
                )
                if resp.status_code != 200 or not resp.text.strip():
                    continue
                data = resp.json()
                if not isinstance(data, list):
                    continue
            except Exception:
                continue

            query_count += 1

            for entry in data:
                if len(entry.get("agentIds", [])) != 5:
                    continue

                agent_ids = sorted(entry["agentIds"])
                agents = [API_AGENT_NAMES.get(aid, f"Unknown({aid})") for aid in agent_ids]
                comp_key = (map_name, *agents)
                is_overall = side is None

                if comp_key in map_seen:
                    existing = seen[comp_key]
                    existing_is_overall = existing["_side"] is None
                    should_update = (
                        (is_overall and not existing_is_overall)
                        or (is_overall == existing_is_overall and entry["rounds"] > existing["rounds"])
                    )
                    if should_update:
                        seen[comp_key].update({
                            "win_rate": float(entry["percentage"]),
                            "wins": entry["wins"],
                            "rounds": entry["rounds"],
                            "_side": side,
                        })
                    continue

                map_seen.add(comp_key)
                seen[comp_key] = {
                    "map": map_name,
                    "agent_1": agents[0],
                    "agent_2": agents[1],
                    "agent_3": agents[2],
                    "agent_4": agents[3],
                    "agent_5": agents[4],
                    "win_rate": float(entry["percentage"]),
                    "wins": entry["wins"],
                    "rounds": entry["rounds"],
                    "_side": side,
                }

        print(f"  {map_name:<10}: {len(map_seen):>4} unique comps  ({query_count} queries)")

    df = pd.DataFrame(seen.values()).drop(columns=["_side"])
    df = df.sort_values(["map", "win_rate"], ascending=[True, False])
    df.to_csv("comp_win_rates.csv", index=False)

    print(f"\nSaved {len(df)} rows to comp_win_rates.csv")
    print("\nPer-map breakdown:")
    print(df.groupby("map").size().to_string())


if __name__ == "__main__":
    main()
