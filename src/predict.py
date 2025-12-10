import os
import joblib
import pandas as pd
from nba_api.stats.static import teams, players

BASE_DIR = os.path.dirname(__file__)
TEAM_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Team.pkl")
PLAYER_MODEL_PATH = os.path.join(BASE_DIR, "RandomForestModel_Player.pkl")
PLAYER_CACHE_PATH = os.path.join(BASE_DIR, "player_cache.csv")
TEAM_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
PLAYER_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
ROLL_FEATURES = [f"{s}_ROLL5" for s in PLAYER_STATS]

team_model = joblib.load(TEAM_MODEL_PATH)
player_model = joblib.load(PLAYER_MODEL_PATH)
player_cache = pd.read_csv(PLAYER_CACHE_PATH, parse_dates=['GAME_DATE'])

nba_teams = {t['full_name'].lower(): t for t in teams.get_teams()}
active_players = players.get_active_players()
player_id_to_name = {p['id']: p['full_name'] for p in active_players}


def get_team_features(team_name):
    team_name_lower = team_name.lower()
    
    matched_team = None
    for t in teams.get_teams():
        if t['full_name'].lower() == team_name_lower or t['nickname'].lower() == team_name_lower:
            matched_team = t
            break
    
    if not matched_team:
        print(f"Warning: Team '{team_name}' not found.")
        return [0] * len(TEAM_STATS)
    
    from nba_api.stats.endpoints import LeagueGameLog
    try:
        log = LeagueGameLog(season='2023-24')
        df = log.get_data_frames()[0]
        
        team_games = df[df['TEAM_ID'] == matched_team['id']]
        
        if team_games.empty:
            return [0] * len(TEAM_STATS)
        
        return [team_games[stat].mean() if stat in team_games.columns else 0 
                for stat in TEAM_STATS]
    except Exception as e:
        print(f"Error fetching team data: {e}")
        return [0] * len(TEAM_STATS)


def predict_top_players(team_names=None, n_players=15, lookback=25):
    results = []
    
    if player_cache.empty:
        print("  ⚠ No player cache data available")
        return results
    
    player_id_col = None
    for col in ['PLAYER_ID', 'Player_ID', 'player_id']:
        if col in player_cache.columns:
            player_id_col = col
            break
    
    if not player_id_col:
        print(f"  ⚠ No PLAYER_ID column found")
        return results
    
    date_col = None
    for col in ['GAME_DATE', 'Game_Date', 'game_date']:
        if col in player_cache.columns:
            date_col = col
            break
    
    if not date_col:
        print(f"  ⚠ No GAME_DATE column found")
        return results
    
    filter_team_abbrev = None
    if team_names:
        filter_team_abbrev = set()
        for team_name in team_names:
            for t in teams.get_teams():
                if t['full_name'].lower() == team_name.lower() or t['nickname'].lower() == team_name.lower():
                    filter_team_abbrev.add(t['abbreviation'])
                    break
    
    grouped = player_cache.groupby(player_id_col)

    for player_id, df_player in grouped:
        if player_id not in player_id_to_name:
            continue
        
        if filter_team_abbrev and 'MATCHUP' in df_player.columns:
            player_teams = set()
            for matchup in df_player['MATCHUP'].unique():
                if isinstance(matchup, str):
                    team_code = matchup.split()[0]
                    player_teams.add(team_code)
            
            if not (player_teams & filter_team_abbrev):
                continue

        df = df_player.sort_values(date_col).copy()

        missing_stats = [s for s in PLAYER_STATS if s not in df.columns]
        if missing_stats:
            continue

        for stat in PLAYER_STATS:
            if stat in df.columns:
                df[f"{stat}_ROLL5"] = (
                    df[stat]
                    .rolling(5, min_periods=1)
                    .mean()
                    .shift(1)
                )

        df = df.dropna(subset=[f"{s}_ROLL5" for s in PLAYER_STATS])

        if len(df) < lookback:
            continue

        recent = df.tail(lookback)

        roll_cols = [f"{s}_ROLL5" for s in PLAYER_STATS]
        if not all(col in recent.columns for col in roll_cols):
            continue

        try:
            features = pd.DataFrame([recent[roll_cols].mean()])
            pred_pts = float(player_model.predict(features)[0])

            avg_stats = {
                stat: round(recent[stat].mean(), 1)
                for stat in PLAYER_STATS
                if stat in recent.columns
            }

            results.append({
                "name": player_id_to_name[player_id],
                "pred_pts": round(pred_pts, 1),
                **avg_stats
            })
        except Exception as e:
            continue

    results.sort(key=lambda x: x["pred_pts"], reverse=True)
    return results[:n_players]


def predict_game(home_team_name, away_team_name):
    home_team = None
    away_team = None
    
    for t in teams.get_teams():
        if t['full_name'].lower() == home_team_name.lower() or t['nickname'].lower() == home_team_name.lower():
            home_team = t
        if t['full_name'].lower() == away_team_name.lower() or t['nickname'].lower() == away_team_name.lower():
            away_team = t
    
    if not home_team or not away_team:
        print("One or both team names are invalid.")
        return

    home_features = get_team_features(home_team_name)
    away_features = get_team_features(away_team_name)
    
    if not home_features or not away_features:
        print("Unable to retrieve team statistics.")
        return

    features = [f"{s}_HOME" for s in TEAM_STATS] + [f"{s}_AWAY" for s in TEAM_STATS]
    X = pd.DataFrame([home_features + away_features], columns=features)
    prob_home = float(team_model.predict_proba(X)[0][1])
    prob_away = 1 - prob_home

    print("\nNBA GAME PREDICTION")
    print(f"{home_team_name.upper()} vs {away_team_name.upper()}")
    print(f"Win Probability: {prob_home*100:.2f}% vs {prob_away*100:.2f}%")
    
    if prob_home > prob_away:
        favorite = home_team_name
    else:
        favorite = away_team_name
    diff = abs(prob_home - prob_away) * 100
    print(f"Favorite: {favorite} by {diff:.2f}%")

    top_players = predict_top_players(team_names=[home_team_name, away_team_name])

    print("\nTOP FANTASY PLAYERS")
    
    if not top_players:
        print("No players found from these teams in the dataset.")
    else:
        print(f"{'RANK':<6}{'NAME':<24}{'PTS':>7}{'REB':>7}{'AST':>7}{'STL':>7}{'BLK':>7}{'TOV':>7}")

        for i, p in enumerate(top_players, 1):
            print(
                f"{i:<6}"
                f"{p['name']:<24} "
                f"{p['pred_pts']:>7.1f}"
                f"{p['REB']:>7.1f} "
                f"{p['AST']:>7.1f} "
                f"{p['STL']:>7.1f} "
                f"{p['BLK']:>7.1f} "
                f"{p['TOV']:>7.1f}"
            )

    print()


if __name__ == "__main__":
    try:
        while True:
            home_team_name = input("Enter Home Team: ").strip()
            away_team_name = input("Enter Away Team: ").strip()
            
            if not home_team_name or not away_team_name:
                print("Both team names are required.\n")
            else:
                predict_game(home_team_name, away_team_name)
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")