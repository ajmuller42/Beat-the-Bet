import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
STATDUMP_DIR = os.path.join(BASE_DIR, "StatDumps")

def parseMinutes(x):
    if isinstance(x, str) and ":" in x:
        minutes, seconds = x.split(":")
        return int(minutes) + int(seconds) / 60
    return 0.0

def loadTeamData(name):
    folder = os.path.join(STATDUMP_DIR, name)
    gameFiles = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]

    games = []
    for x in gameFiles:
        path = os.path.join(folder, x)
        datafile = pd.read_csv(path)
        datafile = datafile[datafile["MP"] != "Did Not Play"]
        datafile["MP"] = datafile["MP"].apply(parseMinutes)
        games.append(datafile)
    
    return games 

def loadAllTeams():
    teams = {}
    for x in os.listdir(STATDUMP_DIR):
        folder = os.path.join(STATDUMP_DIR, x)
        if os.path.isdir(folder):
            teams[x] = loadTeamData(x)
    return teams

if __name__ == "__main__":
    teams = loadAllTeams()
    print("Loaded teams: ", list(teams.keys()))

    for x, y in teams.items():
        print(f"Team: {x}, Number of games loaded: {len(y)}")