import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nba_api.stats.endpoints import LeagueGameLog, PlayerGameLog
from nba_api.stats.static import players
from requests.exceptions import Timeout, ConnectionError

BASE_DIR = os.path.dirname(__file__)

TEAM_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Team.pkl")
PLAYER_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Player.pkl")
PLAYER_CACHE_PATH = os.path.join(BASE_DIR, "player_cache.csv")

TEAM_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
PLAYER_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']

REQUEST_DELAY = 1.0
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0


def last_3_seasons():
    y = datetime.now().year
    return [f"{y-i-1}-{str(y-i)[-2:]}" for i in range(10)]


def safe_api_call(func, *args, **kwargs):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            time.sleep(REQUEST_DELAY)
            return func(*args, **kwargs)
        except (Timeout, ConnectionError) as e:
            if attempt < RETRY_ATTEMPTS - 1:
                print(f"Timeout (attempt {attempt + 1}/{RETRY_ATTEMPTS}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise e


def get_team_data():
    print("Fetching TEAM data...")
    dfs = []
    for season in last_3_seasons():
        try:
            print(f"Fetching season {season}...")
            log = safe_api_call(LeagueGameLog, season=season)
            df = log.get_data_frames()[0]
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            dfs.append(df)
            print(f"{len(df)} games loaded")
        except Exception as e:
            print(f"Error fetching season {season}: {e}")
    
    if not dfs:
        raise RuntimeError("Failed to fetch any team data!")
    
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"Total TEAM rows: {len(final_df)}\n")
    return final_df


def train_team_model():
    df = get_team_data()
    df['HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)

    home_df = df[df['HOME'] == 1][['GAME_ID', 'TEAM_ID'] + TEAM_STATS].copy()
    away_df = df[df['HOME'] == 0][['GAME_ID', 'TEAM_ID'] + TEAM_STATS].copy()

    games = home_df.merge(
        away_df,
        on='GAME_ID',
        suffixes=('_HOME', '_AWAY')
    )
    games['HOME_WIN'] = (games['PTS_HOME'] > games['PTS_AWAY']).astype(int)

    features = [f"{s}_HOME" for s in TEAM_STATS] + [f"{s}_AWAY" for s in TEAM_STATS]
    X = games[features].copy()
    y = games['HOME_WIN'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train.columns = features
    X_test.columns = features
    model = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("="*60)
    print("TEAM MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("="*60 + "\n")

    joblib.dump(model, TEAM_MODEL_PATH)
    print("✓ Team model saved.\n")


def get_player_data():
    if os.path.exists(PLAYER_CACHE_PATH):
        print("Loading cached PLAYER data...")
        df = pd.read_csv(PLAYER_CACHE_PATH, parse_dates=['GAME_DATE'])
        print(f"Loaded {len(df)} rows from cache.\n")
        return df

    print("Fetching PLAYER data (first run, will take a while)...\n")
    dfs = []

    active_players_list = players.get_active_players()
    print(f"Active players: {len(active_players_list)}")
    print(f"Fetching data for first 450 players...\n")

    failed_players = []
    
    num_players = 450

    for idx, player in enumerate(active_players_list[:num_players]): 
        player_name = player['full_name']
        print(f"[{idx+1:3d}/num_players] {player_name:<25}", end="", flush=True)

        seasons_success = 0
        for season in last_3_seasons():
            try:
                log = safe_api_call(PlayerGameLog, player_id=player['id'], season=season)
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
                seasons_success += 1
            except (Timeout, ConnectionError) as e:
                print(f"\nTimeout for {player_name} - {season}, skipping...")
                failed_players.append((player_name, season))
            except Exception as e:
                pass
        print(f"({seasons_success}/3 seasons)")

    if failed_players:
        print(f"\nFailed to fetch {len(failed_players)} player-season combinations (network timeouts)")

    if not dfs:
        raise RuntimeError("No player data fetched!")

    final_df = pd.concat(dfs, ignore_index=True)
    final_df['TEAM_ID'] = final_df['TEAM_ID'].astype('Int64')

    print(f"\nTotal PLAYER rows: {len(final_df)}")
    final_df.to_csv(PLAYER_CACHE_PATH, index=False)
    print("Player cache written successfully.\n")
    return final_df


def train_player_model():
    df = get_player_data()
    
    print("Preparing player data for model training...")
    print(f"Initial rows: {len(df)}")
    print(f"Available columns: {df.columns.tolist()}\n")
    
    required_cols = ['PLAYER_ID', 'GAME_DATE'] + PLAYER_STATS
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
    
    if df.empty:
        print("No player data available!")
        return
    
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    
    print("Creating 5-game rolling averages...")
    for stat in PLAYER_STATS:
        if stat in df.columns:
            df[f"{stat}_ROLL5"] = df.groupby('PLAYER_ID')[stat].shift(1).rolling(5, min_periods=1).mean()
        else:
            print(f"Column {stat} not found, filling with 0")
            df[stat] = 0
            df[f"{stat}_ROLL5"] = 0

    df_clean = df.dropna(subset=[f"{s}_ROLL5" for s in PLAYER_STATS] + ['PTS'])
    print(f"Rows after removing NaN: {len(df_clean)}")
    
    if len(df_clean) < 10:
        print("Insufficient data for training (need at least 10 samples)")
        return
    
    X = df_clean[[f"{s}_ROLL5" for s in PLAYER_STATS]].copy()
    y = df_clean['PTS'].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}\n")
    
    test_size = min(0.2, max(0.1, 10 / len(df_clean)))
    print(f"Using test_size={test_size:.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")
    
    model = RandomForestRegressor(n_estimators=120, max_depth=8, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("="*60)
    print("PLAYER MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"R² Score:              {r2:.4f}")
    print(f"Mean Absolute Error:   {mae:.4f} points")
    print(f"Root Mean Squared Err: {rmse:.4f} points")
    print(f"Mean Squared Error:    {mse:.4f}")
    print("="*60 + "\n")

    joblib.dump(model, PLAYER_MODEL_PATH)
    print("Player model saved.\n")


if __name__ == "__main__":
    try:
        train_team_model()
        train_player_model()
        print("All models trained successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise