import os
import pandas as pd
import random
from .parse_StatDumps import parse_StatDumps

STATDUMPS_DIR = "StatDumps"


def load_all_teams():
    teams = {}

    if not os.path.exists(STATDUMPS_DIR):
        error_message = "{} folder not found!".format(STATDUMPS_DIR)
        raise FileNotFoundError(error_message)

    all_items = os.listdir(STATDUMPS_DIR)
    for team in all_items:
        team_path = os.path.join(STATDUMPS_DIR, team)
        if not os.path.isdir(team_path):
            continue

        csv_files = []
        team_files = os.listdir(team_path)
        for file in team_files:
            if file.endswith(".csv"):
                csv_files.append(file)

        if not csv_files:
            print("No CSVs for team {}".format(team))
            continue

        csv_files.sort()
        latest = csv_files[-1]
        teams[team] = os.path.join(team_path, latest)

    return teams


def buildDataset():
    teams = load_all_teams()
    team_list = list(teams.keys())
    print("Loaded teams: {}".format(team_list))

    rows = []

    default_stats = {
        'ORtg': 110,
        'DRtg': 105,
        'TS%': 0.55,
        'AST%': 25,
        'TOV%': 12
    }

    for teamA in teams:
        fileA = teams[teamA]
        for teamB in teams:
            fileB = teams[teamB]
            if teamA == teamB:
                continue

            try:
                statsA = parse_StatDumps(fileA)
                statsB = parse_StatDumps(fileB)
            except Exception as e:
                error_msg = "Skipping matchup {} vs {}: {}".format(teamA, teamB, e)
                print(error_msg)
                continue

            for key in default_stats:
                default_value = default_stats[key]
                
                if key in statsA:
                    statsA[key] = float(statsA[key])
                else:
                    statsA[key] = float(default_value)
                
                if key in statsB:
                    statsB[key] = float(statsB[key])
                else:
                    statsB[key] = float(default_value)

            result_label = random.choice([0, 1])

            row = {
                "teamA": teamA,
                "teamB": teamB,
                "offensiveRatingDifference": statsA['ORtg'] - statsB['ORtg'],
                "defensiveRatingDifference": statsA['DRtg'] - statsB['DRtg'],
                "totalScoreDifference": statsA['TS%'] - statsB['TS%'],
                "assistDifference": statsA['AST%'] - statsB['AST%'],
                "turnoverDifference": statsA['TOV%'] - statsB['TOV%'],
                "result": result_label
            }

            rows.append(row)

    if not rows:
        print("No dataset rows generated. Fix your CSV files.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("dataset.csv", index=False)
    print("Dataset built with {} rows and saved to dataset.csv!".format(len(df)))


if __name__ == "__main__":
    buildDataset()