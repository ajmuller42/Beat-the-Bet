from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import pandas as pd
import json
from nba_api.stats.static import teams, players
from src.predict import predict_game, get_team_features, predict_top_players

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models + cache automatically
TEAM_MODEL_PATH = os.path.join("src", "RandomForestModel_Team.pkl")
PLAYER_MODEL_PATH = os.path.join("src", "RandomForestModel_Player.pkl")
PLAYER_CACHE_PATH = os.path.join("src", "player_cache.csv")

team_model = joblib.load(TEAM_MODEL_PATH)
player_model = joblib.load(PLAYER_MODEL_PATH)
player_cache = pd.read_csv(PLAYER_CACHE_PATH, parse_dates=['GAME_DATE'])

@app.get("/teams")
def get_teams():
    """Return a list of NBA teams."""
    team_list = [t["full_name"] for t in teams.get_teams()]
    return {"teams": sorted(team_list)}

@app.get("/predict")
def api_predict(home: str, away: str):
    """Return prediction + top players in JSON."""
    try:
        # Team win probability
        home_features = get_team_features(home)
        away_features = get_team_features(away)

        import pandas as pd
        TEAM_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        feature_cols = [f"{s}_HOME" for s in TEAM_STATS] + [f"{s}_AWAY" for s in TEAM_STATS]

        X = pd.DataFrame([home_features + away_features], columns=feature_cols)
        prob_home = float(team_model.predict_proba(X)[0][1])
        prob_away = 1 - prob_home

        # Player predictions
        top_players = predict_top_players(team_names=[home, away], n_players=10)

        return {
            "home_team": home,
            "away_team": away,
            "prob_home": prob_home,
            "prob_away": prob_away,
            "players": top_players,
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/stats")
def get_stats():
    stats_path = os.path.join("src", "model_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            return json.load(f)
    else:
        return {"error": "Model statistics not found."}
