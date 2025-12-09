import pandas as pd

NUMERIC_COLS = ['ORtg', 'DRtg', 'TS%', 'AST%', 'TOV%', 'USG%', 'BPM']

def mp_to_minutes(mp):
    if isinstance(mp, str) and ":" in mp:
        m, s = mp.split(":")
        return int(m) + int(s) / 60
    return 0.0


def parse_StatDumps(filepath):
    try:
        df = pd.read_csv(filepath)
    except:
        df = pd.read_csv(filepath, sep="\t")

    df.columns = [c.strip() for c in df.columns]
    name_col = df.columns[0]

    totals = df[df[name_col].str.contains("Team Totals", na=False)]

    team_stats = {}
    if not totals.empty:
        row = totals.iloc[0]
        for col in ['ORtg', 'DRtg', 'TS%', 'AST%', 'TOV%']:
            team_stats[col] = float(row[col]) if col in row and pd.notna(row[col]) else 0.0

    players_df = df[
        (~df[name_col].str.contains("Team Totals", na=False)) &
        (~df["MP"].astype(str).str.contains("Did Not", na=False))
    ].copy()

    players_df["MP_min"] = players_df["MP"].apply(mp_to_minutes)
    players_df = players_df[players_df["MP_min"] >= 10]  

    for col in NUMERIC_COLS:
        if col in players_df.columns:
            players_df[col] = pd.to_numeric(players_df[col], errors="coerce").fillna(0.0)
        else:
            players_df[col] = 0.0

    players = []
    for _, row in players_df.iterrows():
        players.append({
            "name": row[name_col],
            "MP": row["MP_min"],
            "ORtg": row["ORtg"],
            "DRtg": row["DRtg"],
            "TS%": row["TS%"],
            "AST%": row["AST%"],
            "TOV%": row["TOV%"],
            "USG%": row["USG%"],
            "BPM": row["BPM"]
        })
    if not team_stats and not players_df.empty:
        total_minutes = players_df["MP_min"].sum()
        for col in ['ORtg', 'DRtg', 'TS%', 'AST%', 'TOV%']:
            team_stats[col] = float(
                (players_df[col] * players_df["MP_min"]).sum() / total_minutes
            )

    return {
        "team_stats": team_stats,
        "players": players
    }
