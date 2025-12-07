import joblib
import re
import os
from .dataLoader import loadAllTeams
from .parse_StatDumps import parse_StatDumps

BASE_DIR = os.path.dirname(__file__)  
model_path = os.path.join(BASE_DIR, 'model.pkl')
model = joblib.load(model_path)

def extractTeam(filename):
    return re.split(r"\d", filename)[0]

def get_latest_csv(team_name, statdumps_dir="StatDumps"):
    team_dir = os.path.join(statdumps_dir, team_name)
    if not os.path.exists(team_dir):
        raise FileNotFoundError(f"No folder for team {team_name}")
    csvs = [f for f in os.listdir(team_dir) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSV files found for team {team_name}")
    return os.path.join(team_dir, sorted(csvs)[-1])

def predict_matchup(team1_name, team2_name):
    team1_csv = get_latest_csv(team1_name)
    team2_csv = get_latest_csv(team2_name)

    team1_stats = parse_StatDumps(team1_csv)
    team2_stats = parse_StatDumps(team2_csv)

    features = [[
        team1_stats['ORtg'] - team2_stats['ORtg'],
        team1_stats['DRtg'] - team2_stats['DRtg'],
        team1_stats['TS%'] - team2_stats['TS%'],
        team1_stats['AST%'] - team2_stats['AST%'],
        team1_stats['TOV%'] - team2_stats['TOV%']
    ]]

    probability = model.predict_proba(features)[0]

    return {
        team1_name: float(probability[1]),
        team2_name: float(probability[0])
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 -m src.predict <team1> <team2>")
        sys.exit(1)

    team1 = sys.argv[1]
    team2 = sys.argv[2]

    result = predict_matchup(team1, team2)
    print(f"Predicted probabilities:\n{result}")
