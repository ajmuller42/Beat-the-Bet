import os
import joblib
import pandas as pd
from nba_api.stats.static import teams, players

BASE_DIR = os.path.dirname(__file__)
TEAM_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Team.pkl")
PLAYER_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Player.pkl")
PLAYER_CACHE_PATH = os.path.join(BASE_DIR, "player_cache.csv")
PLAYER_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']

team_model = joblib.load(TEAM_MODEL_PATH)
player_model = joblib.load(PLAYER_MODEL_PATH)
player_cache = pd.read_csv(PLAYER_CACHE_PATH, parse_dates=['GAME_DATE'])

nba_teams = {t['full_name'].lower(): t for t in teams.get_teams()}
active_players = players.get_active_players()
player_id_map = {p['id']: p['full_name'] for p in active_players}
player_team_map = {p['full_name'].lower(): p.get('team_name', '') for p in active_players}

def get_team_features(team_name):
    df = player_cache.copy()
    df = df[df['Player_ID'].isin([p['id'] for p in active_players if p.get('team_name', '').lower() == team_name.lower()])]
    if df.empty:
        return [0] * len(PLAYER_STATS)
    return [df[s].mean() if s in df.columns else 0 for s in PLAYER_STATS]

def predict_top_players(team_name, n_players=5):
    team_players = [p for p in active_players if p.get('team_name', '').lower() == team_name.lower()]
    predictions = []
    for p in team_players:
        pid = p['id']
        df = player_cache[player_cache['Player_ID'] == pid].sort_values('GAME_DATE', ascending=False).head(5)
        stats = [df[s].mean() if s in df.columns and not df.empty else 0 for s in PLAYER_STATS]
        pred_pts = float(player_model.predict([stats])[0])
        predictions.append((p['full_name'], pred_pts))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n_players]

def predict_game(home_team_name, away_team_name):
    home_team = nba_teams.get(home_team_name.lower())
    away_team = nba_teams.get(away_team_name.lower())
    if not home_team or not away_team:
        print("One or both team names are invalid.")
        return

    home_features = get_team_features(home_team_name)
    away_features = get_team_features(away_team_name)
    X = [home_features + away_features]
    prob_home = float(team_model.predict_proba(X)[0][1])
    prob_away = 1 - prob_home

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"NBA Game Prediction: {home_team_name} vs {away_team_name}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Win Probability: {home_team_name}: {prob_home*100:.2f}% | {away_team_name}: {prob_away*100:.2f}%\n")

    top_home = predict_top_players(home_team_name)
    top_away = predict_top_players(away_team_name)

    print(f"Top Home Players to Watch ({home_team_name}):")
    print("──────────────────────────────────────────────────")
    for name, pts in top_home:
        print(f"{name:<25} → Predicted Points: {pts:.1f}")
    print()
    print(f"Top Away Players to Watch ({away_team_name}):")
    print("──────────────────────────────────────────────────")
    for name, pts in top_away:
        print(f"{name:<25} → Predicted Points: {pts:.1f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if __name__ == "__main__":
    home_team_name = input("Enter Home Team: ")
    away_team_name = input("Enter Away Team: ")
    predict_game(home_team_name, away_team_name)
