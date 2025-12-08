import joblib
import os
from .parse_StatDumps import parse_StatDumps

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(model_path)

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
    stats = parse_StatDumps(filepath)
    cleaned = {}
    
    for key in DEFAULT_STATS:
        default_value = DEFAULT_STATS[key]
        
        if key in stats:
            stat_value = stats[key]
        else:
            stat_value = default_value
        
        cleaned[key] = float(stat_value)

    return cleaned


def predict_matchup(team1_name, team2_name):
    team1_csv = get_latest_csv(team1_name)
    team2_csv = get_latest_csv(team2_name)

    team1_stats = load_team_stats(team1_csv)
    team2_stats = load_team_stats(team2_csv)

    ortg_diff = team1_stats['ORtg'] - team2_stats['ORtg']
    drtg_diff = team1_stats['DRtg'] - team2_stats['DRtg']
    ts_diff = team1_stats['TS%'] - team2_stats['TS%']
    ast_diff = team1_stats['AST%'] - team2_stats['AST%']
    tov_diff = team1_stats['TOV%'] - team2_stats['TOV%']

    features = [[ortg_diff, drtg_diff, ts_diff, ast_diff, tov_diff]]

    prob = model.predict_proba(features)[0]

    result = {}
    result[team1_name] = float(prob[1])
    result[team2_name] = float(prob[0])

    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python3 -m src.predict <team1> <team2>")
        sys.exit(1)

    team1 = sys.argv[1]
    team2 = sys.argv[2]

    result = predict_matchup(team1, team2)
    
    print("\nPredicted probabilities:")
    print(result)