# scrape_data.py — Scrape per-agent win rates from rib.gg → agent_win_rates.csv

import requests
import pandas as pd

from constants import API_AGENT_NAMES, API_MAP_IDS

HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.rib.gg",
    "Referer": "https://www.rib.gg/analytics/agents",
}


def main():
    rows = []
    for map_api_id, map_name in API_MAP_IDS.items():
        resp = requests.get(
            "https://be-prod.rib.gg/v1/x/agent-win-rates",
            params={"mapId": map_api_id},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        for entry in resp.json():
            agent_name = API_AGENT_NAMES.get(entry["agentId"])
            if agent_name:
                rows.append({
                    "agent": agent_name,
                    "map": map_name,
                    "win_rate": float(entry["percentage"]),
                    "wins": entry["wins"],
                    "rounds": entry["rounds"],
                })
        print(f"Fetched {map_name}")

    df = pd.DataFrame(rows)

    # Fill missing agent/map combinations with zeros
    all_agents = sorted(df["agent"].unique())
    all_maps = sorted(df["map"].unique())
    full_index = pd.MultiIndex.from_product([all_agents, all_maps], names=["agent", "map"])
    df = (
        df.set_index(["agent", "map"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["map", "win_rate"], ascending=[True, False])
    )

    df.to_csv("agent_win_rates.csv", index=False)
    print(f"\nSaved {len(df)} rows to agent_win_rates.csv")


if __name__ == "__main__":
    main()