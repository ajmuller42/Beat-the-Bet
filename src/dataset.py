import pandas as pd
import random
from .dataLoader import loadAllTeams  # Make sure this file is in src/

def buildDataset():
    # Load all teams and their games into memory
    teams = loadAllTeams()
    print(f"Loaded teams: {list(teams.keys())}")

    rows = []

    # Loop through all possible matchups
    for teamA, gamesA in teams.items():
        for teamB, gamesB in teams.items():
            if teamA == teamB:
                continue

            # Take pairings of games (simple: last game for each team)
            gameA = gamesA[-1]  # use most recent game for example
            gameB = gamesB[-1]

            # Ensure stats exist, fill defaults if missing
            default_stats = {'ORtg': 110, 'DRtg': 105, 'TS%': 0.55, 'AST%': 25, 'TOV%': 12}
            for key in default_stats:
                default_stats = {'ORtg': 110, 'DRtg': 105, 'TS%': 0.55, 'AST%': 25, 'TOV%': 12}
                gameA[key] = pd.to_numeric(gameA.get(key, pd.Series([default_stats[key]]*len(gameA))), errors='coerce')
                gameB[key] = pd.to_numeric(gameB.get(key, pd.Series([default_stats[key]]*len(gameB))), errors='coerce')

                gameA[key] = gameA[key].fillna(default_stats[key])
                gameB[key] = gameB[key].fillna(default_stats[key])

                if key not in gameA.columns:
                    gameA[key] = default_stats[key]
                if key not in gameB.columns:
                    gameB[key] = default_stats[key]

            row = {
                "date": "unknown",  # no date field available directly
                "teamA": teamA,
                "teamB": teamB,
                "offensiveRatingDifference": gameA['ORtg'].mean() - gameB['ORtg'].mean(),
                "defensiveRatingDifference": gameA['DRtg'].mean() - gameB['DRtg'].mean(),
                "totalScoreDifference": gameA['TS%'].mean() - gameB['TS%'].mean(),
                "assistDifference": gameA['AST%'].mean() - gameB['AST%'].mean(),
                "turnoverDifference": gameA['TOV%'].mean() - gameB['TOV%'].mean(),
                "result": random.choice([0, 1])  # temporary random result for training
            }
            rows.append(row)

    if not rows:
        print("No rows generated — check your CSV files!")
        return

    df = pd.DataFrame(rows)
    df.to_csv("dataset.csv", index=False)
    print(f"Dataset built with {len(df)} rows and saved to dataset.csv!")

if __name__ == "__main__":
    buildDataset()
