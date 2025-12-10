import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nba_api.stats.endpoints import LeagueGameLog, PlayerGameLog
from nba_api.stats.static import players

BASE_DIR = os.path.dirname(__file__)

TEAM_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Team.pkl")
PLAYER_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Player.pkl")
PLAYER_CACHE_PATH = os.path.join(BASE_DIR, "player_cache.csv")

TEAM_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
PLAYER_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']


def last_3_seasons():
    y = datetime.now().year
    return [f"{y-i-1}-{str(y-i)[-2:]}" for i in range(3)]


def get_team_data():
    print("Fetching TEAM data...")
    dfs = []
    for season in last_3_seasons():
        log = LeagueGameLog(season=season)
        df = log.get_data_frames()[0]
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"Total TEAM rows: {len(final_df)}")
    return final_df


def train_team_model():
    df = get_team_data()
    df['HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)

    home_df = df[df['HOME'] == 1]
    away_df = df[df['HOME'] == 0]

    games = home_df.merge(
        away_df,
        on='GAME_ID',
        suffixes=('_HOME', '_AWAY')
    )
    games['HOME_WIN'] = (games['PTS_HOME'] > games['PTS_AWAY']).astype(int)

    features = [f"{s}_HOME" for s in TEAM_STATS] + [f"{s}_AWAY" for s in TEAM_STATS]
    X = games[features]
    y = games['HOME_WIN']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print("TEAM MODEL ACCURACY:", round(acc, 3))
    joblib.dump(model, TEAM_MODEL_PATH)
    print("Team model saved.")


def get_player_data():
    if os.path.exists(PLAYER_CACHE_PATH):
        print("\nLoading cached PLAYER data...")
        df = pd.read_csv(PLAYER_CACHE_PATH, parse_dates=['GAME_DATE'])
        print(f"Loaded {len(df)} rows from cache.")
        print("Columns:", df.columns.tolist())
        return df

    print("\nFetching PLAYER data (first run, will take a while)...")
    dfs = []

    active_players = players.get_active_players()
    print(f"Active players: {len(active_players)}")

    for idx, player in enumerate(active_players[:120]): 
        print(f"Fetching {player['full_name']} ({idx+1}/120)")

        for season in last_3_seasons():
            try:
                log = PlayerGameLog(player_id=player['id'], season=season)
                df_player = log.get_data_frames()[0]

                df_player.columns = [c.upper() for c in df_player.columns]

                if 'PLAYER_ID' not in df_player.columns:
                    df_player['PLAYER_ID'] = player['id']
                if 'TEAM_ID' not in df_player.columns:
                    if 'TEAM_ID_TEAM' in df_player.columns:
                        df_player['TEAM_ID'] = df_player['TEAM_ID_TEAM']
                    else:
                        df_player['TEAM_ID'] = np.nan

                df_player['SEASON'] = season
                df_player['GAME_DATE'] = pd.to_datetime(df_player['GAME_DATE'])

                dfs.append(df_player)
            except Exception as e:
                print(f"  ERROR fetching {player['full_name']} {season}: {e}")

    if not dfs:
        raise RuntimeError("No player data fetched!")

    final_df = pd.concat(dfs, ignore_index=True)

    final_df['TEAM_ID'] = final_df['TEAM_ID'].astype('Int64')

    print(f"Total PLAYER rows: {len(final_df)}")
    print(f"Columns: {final_df.columns.tolist()}")

    final_df.to_csv(PLAYER_CACHE_PATH, index=False)
    print("Player cache written successfully.\n")
    return final_df



def train_player_model():
    df = get_player_data()
    df = df.sort_values(['Player_ID', 'GAME_DATE'])

    for stat in PLAYER_STATS:
        df[f"{stat}_ROLL5"] = df.groupby('Player_ID')[stat].shift(1).rolling(5, min_periods=1).mean()

    df = df.dropna()
    X = df[[f"{s}_ROLL5" for s in PLAYER_STATS]]
    y = df['PTS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, max_depth=8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    corr = np.corrcoef(model.predict(X_test), y_test)[0, 1]
    print("PLAYER MODEL CORRELATION:", round(corr, 3))
    joblib.dump(model, PLAYER_MODEL_PATH)
    print("Player model saved.")


if __name__ == "__main__":
    train_team_model()
    train_player_model()
