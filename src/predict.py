import os
import joblib
import pandas as pd
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import LeagueGameLog

BASE_DIR = os.path.dirname(__file__)
TEAM_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Team.pkl")
PLAYER_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Player.pkl")
PLAYER_CACHE_PATH = os.path.join(BASE_DIR, "player_cache.csv")
TEAM_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
PLAYER_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
ROLL_FEATURES = [f"{s}_ROLL5" for s in PLAYER_STATS]

team_model = joblib.load(TEAM_MODEL_PATH)
print(team_model.feature_names_in_)
player_model = joblib.load(PLAYER_MODEL_PATH)
player_cache = pd.read_csv(PLAYER_CACHE_PATH, parse_dates=['GAME_DATE'])

nba_teams = {t['full_name'].lower(): t for t in teams.get_teams()}
active_players = players.get_active_players()
player_id_to_name = {p['id']: p['full_name'] for p in active_players}

def get_team_roll_features(team_name, season='2024-25', lookback=5):
    team_id = None

    # ✅ iterate over VALUES, not keys
    for t in nba_teams.values():
        if (
            t['full_name'].lower() == team_name.lower()
            or t['nickname'].lower() == team_name.lower()
        ):
            team_id = t['id']
            break  # break ONLY when found

    if team_id is None:
        raise ValueError(f"Team '{team_name}' not found")

    log = LeagueGameLog(season=season)
    df = log.get_data_frames()[0]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    team_games = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE')

    if len(team_games) < lookback:
        raise ValueError(f"Not enough games for {team_name}")

    recent = team_games.tail(lookback)

    return [
        float(recent['PTS'].mean()),
        float(recent['REB'].mean()),
        float(recent['AST'].mean()),
        float(recent['STL'].mean()),
        float(recent['BLK'].mean()),
        float(recent['TOV'].mean()),
    ]



def predict_top_players(team_names=None, n_players=15, lookback=25):
    results = []

    if player_cache.empty:
        return results

    grouped = player_cache.groupby("PLAYER_ID")

    for player_id, df_player in grouped:
        if player_id not in player_id_to_name:
            continue

        df = df_player.sort_values("GAME_DATE").copy()

        missing = [s for s in PLAYER_STATS if s not in df.columns]
        if missing:
            continue

        for stat in PLAYER_STATS:
            df[f"{stat}_ROLL5"] = (
                df[stat]
                .rolling(5, min_periods=1)
                .mean()
                .shift(1)
            )

        df = df.dropna(subset=ROLL_FEATURES)
        if len(df) < lookback:
            continue

        recent = df.tail(lookback)

        features = pd.DataFrame([recent[ROLL_FEATURES].iloc[-1]])

        pred_pts = float(player_model.predict(features)[0])

        results.append({
            "name": player_id_to_name[player_id],
            "pred_pts": round(pred_pts, 1),
            **{stat: round(recent[stat].mean(), 1) for stat in PLAYER_STATS}
        })

    results.sort(key=lambda x: x["pred_pts"], reverse=True)
    return results[:n_players]


def predict_game(home_team_name, away_team_name):
    home_features = get_team_roll_features(home_team_name)
    away_features = get_team_roll_features(away_team_name)


    # ### CHANGED ### explicit feature schema
    columns = (
        [f"{c}_HOME" for c in ROLL_FEATURES] +
        [f"{c}_AWAY" for c in ROLL_FEATURES]
    )


    X = pd.DataFrame([home_features + away_features], columns=columns)


    prob_home = float(team_model.predict_proba(X)[0][1])
    prob_away = 1 - prob_home


    print("\nNBA GAME PREDICTION")
    print(f"{home_team_name.upper()} vs {away_team_name.upper()}")
    print(f"Win Probability: {prob_home*100:.2f}% vs {prob_away*100:.2f}%")


    favorite = home_team_name if prob_home > prob_away else away_team_name
    diff = abs(prob_home - prob_away) * 100
    print(f"Favorite: {favorite} by {diff:.2f}%")


    top_players = predict_top_players(team_names=[home_team_name, away_team_name])


    print("\nTOP FANTASY PLAYERS")
    print(f"{'RANK':<6}{'NAME':<24}{'PTS':>7}{'REB':>7}{'AST':>7}{'STL':>7}{'BLK':>7}{'TOV':>7}")


    for i, p in enumerate(top_players, 1):
        print(
            f"{i:<6}"
            f"{p['name']:<24}"
            f"{p['pred_pts']:>7.1f}"
            f"{p['REB']:>7.1f}"
            f"{p['AST']:>7.1f}"
            f"{p['STL']:>7.1f}"
            f"{p['BLK']:>7.1f}"
            f"{p['TOV']:>7.1f}"
    )
        
if __name__ == '__main__':
    try:
        while True:
            home = input("Enter Home Team: ").strip()
            away = input("Enter Away Team: ").strip()


            if home and away:
                predict_game(home, away)
            else:
                print("Both team names are required.\n")
    except KeyboardInterrupt:
        print("\nExiting...")