import pandas as pd

def parse_StatDumps(filepath):
    try:
        datafile = pd.read_csv(filepath)
    except:
        datafile = pd.read_csv(filepath, sep="\t")

    datafile.columns = [col.strip() for col in datafile.columns]

    first_col = datafile.columns[0]

    totals = datafile[datafile[first_col].str.contains('Team Totals', na=False)]
    numeric_cols = ['ORtg','DRtg','TS%','AST%','TOV%']

    if not totals.empty:
        totals_row = totals.iloc[0]
        return {k: float(totals_row[k]) if k in totals_row else 0.0 for k in numeric_cols}

    players = datafile[datafile[first_col] != 'Did Not Play']
    for col in numeric_cols:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors='coerce')
        else:
            players[col] = 0.0

    return {k: float(players[k].mean()) for k in numeric_cols}
