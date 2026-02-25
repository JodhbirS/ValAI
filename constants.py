# constants.py — Agent, map, and role identity mappings.

AGENTS: list[str] = sorted([
    "Astra", "Breach", "Brimstone", "Chamber", "Clove", "Cypher",
    "Deadlock", "Fade", "Gekko", "Harbor", "Iso", "Jett", "KAY/O",
    "Killjoy", "Neon", "Omen", "Phoenix", "Raze", "Reyna", "Sage",
    "Skye", "Sova", "Tejo", "Veto", "Viper", "Vyse", "Waylay", "Yoru",
])

MAPS: list[str] = sorted([
    "Abyss", "Ascent", "Bind", "Breeze", "Corrode", "Fracture",
    "Haven", "Icebox", "Lotus", "Pearl", "Split", "Sunset",
])

AGENT_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(AGENTS)}
IDX_TO_AGENT: dict[int, str] = {i: a for i, a in enumerate(AGENTS)}

MAP_TO_IDX: dict[str, int] = {m: i for i, m in enumerate(MAPS)}
IDX_TO_MAP: dict[int, str] = {i: m for i, m in enumerate(MAPS)}

NUM_AGENTS: int = len(AGENTS)
NUM_MAPS: int = len(MAPS)

# Role index: 0=Controller, 1=Duelist, 2=Initiator, 3=Sentinel
AGENT_ROLES: dict[str, int] = {
    "Astra": 0, "Brimstone": 0, "Clove": 0,
    "Harbor": 0, "Omen": 0, "Viper": 0,
    "Iso": 1, "Jett": 1, "Neon": 1, "Phoenix": 1,
    "Raze": 1, "Reyna": 1, "Waylay": 1, "Yoru": 1,
    "Breach": 2, "Fade": 2, "Gekko": 2, "KAY/O": 2,
    "Skye": 2, "Sova": 2, "Tejo": 2,
    "Chamber": 3, "Cypher": 3, "Deadlock": 3,
    "Killjoy": 3, "Sage": 3, "Veto": 3, "Vyse": 3,
}

ROLE_NAMES: dict[int, str] = {
    0: "Controller",
    1: "Duelist",
    2: "Initiator",
    3: "Sentinel",
}

NUM_ROLES: int = 4

# rib.gg API mappings (used by scrape_data.py / scrape_comps.py)

API_AGENT_NAMES: dict[int, str] = {
    1: "Breach", 2: "Raze", 3: "Cypher", 4: "Sova",
    5: "Killjoy", 6: "Viper", 7: "Sage", 8: "Brimstone",
    9: "Phoenix", 10: "Skye", 11: "Omen", 12: "Jett",
    13: "Reyna", 14: "Yoru", 15: "Astra", 16: "KAY/O",
    17: "Chamber", 18: "Neon", 19: "Fade", 20: "Harbor",
    21: "Gekko", 22: "Deadlock", 23: "Iso", 25: "Clove",
    26: "Vyse", 27: "Tejo", 28: "Waylay", 29: "Veto",
}

API_MAP_IDS: dict[int, str] = {
    1: "Ascent", 2: "Split", 3: "Bind", 4: "Fracture",
    7: "Haven", 8: "Pearl", 9: "Breeze", 10: "Icebox",
    11: "Lotus", 12: "Sunset", 13: "Abyss", 14: "Corrode",
}

API_REGIONS: dict[int, str] = {
    1: "EMEA", 2: "NA", 3: "LATAM", 4: "Brazil", 5: "APAC", 7: "Korea",
}
