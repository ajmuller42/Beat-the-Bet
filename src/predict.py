import joblib
import os
from .parse_StatDumps import parse_StatDumps

BASE_DIR = os.path.dirname(__file__)
XGBoostModel_path = os.path.join(BASE_DIR, "RandomForestModel.pkl")
XGBoostModel = joblib.load(XGBoostModel_path)

STATDUMPS_DIR = "StatDumps"

DEFAULT_STATS = {
    'ORtg': 110,
    'DRtg': 105,
    'TS%': 0.55,
    'AST%': 25,
    'TOV%': 12
}


def get_latest_csv(team_name):
    team_dir = os.path.join(STATDUMPS_DIR, team_name)

    if not os.path.exists(team_dir):
        error_message = "No folder found for team '{}'".format(team_name)
        raise FileNotFoundError(error_message)

    csv_files = []
    all_files = os.listdir(team_dir)
    for file in all_files:
        if file.endswith(".csv"):
            csv_files.append(file)

    if not csv_files:
        error_message = "No CSV files found for team '{}'".format(team_name)
        raise FileNotFoundError(error_message)

    csv_files.sort()
    latest_file = csv_files[-1]
    return os.path.join(team_dir, latest_file)


def load_team_stats(filepath):
    parsed = parse_StatDumps(filepath)

    team_stats = parsed["team_stats"]
    players = parsed["players"]

    cleaned = {}
    for key, default_value in DEFAULT_STATS.items():
        value = team_stats.get(key, default_value)
        cleaned[key] = float(value)

    return cleaned, players



def predict_matchup(team1_name, team2_name):
    team1_csv = get_latest_csv(team1_name)
    team2_csv = get_latest_csv(team2_name)

    team1_stats, team1_players = load_team_stats(team1_csv)
    team2_stats, team2_players = load_team_stats(team2_csv)

    ortg_diff = team1_stats['ORtg'] - team2_stats['ORtg']
    drtg_diff = team1_stats['DRtg'] - team2_stats['DRtg']
    ts_diff   = team1_stats['TS%']  - team2_stats['TS%']
    ast_diff  = team1_stats['AST%'] - team2_stats['AST%']
    tov_diff  = team1_stats['TOV%'] - team2_stats['TOV%']

    features = [[ortg_diff, drtg_diff, ts_diff, ast_diff, tov_diff]]
    prob = XGBoostModel.predict_proba(features)[0]

    print("\nPredicted probabilities:")
    print(f"{team1_name}: {prob[1]:.3f}")
    print(f"{team2_name}: {prob[0]:.3f}")

    print("\nTeam stat comparison:")
    for k in team1_stats:
        print(
            f"{k}: "
            f"{team1_name} {team1_stats[k]:.2f} | "
            f"{team2_name} {team2_stats[k]:.2f}"
        )

    print("\nKey players to watch:\n")

    print(team1_name)
    for p in get_key_players(team1_players):
        print(" -", format_player(p))

    print("\n" + team2_name)
    for p in get_key_players(team2_players):
        print(" -", format_player(p))

    return {
        team1_name: float(prob[1]),
        team2_name: float(prob[0])
    }



def format_player(p):
    return (
        f"{p['name']} | "
        f"MP {p['MP']:.1f} | "
        f"BPM {p['BPM']:+.1f} | "
        f"USG {p['USG%']:.1f}% | "
        f"TS {p['TS%']*100:.1f}%"
    )

def get_key_players(players, n=3):
    ranked = []

    for p in players:
        impact = (
            0.4 * p["BPM"] +
            0.3 * p["USG%"] -
            0.2 * p["DRtg"] +
            0.1 * p["ORtg"]
        ) * (p["MP"] / 36)

        ranked.append((impact, p))

    ranked.sort(reverse=True, key=lambda X: X[0])
    return [p for _, p in ranked[:n]]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 -m src.predict <team1> <team2>")
        sys.exit(1)

    team1 = sys.argv[1]
    team2 = sys.argv[2]

    predict_matchup(team1, team2)
