from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import TeamGameLogs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

player_dict = players.get_players()

# Use ternary operator or write function 
# Names are case sensitive
bron = [player for player in player_dict if player['full_name'] == 'LeBron James'][0]
bron_id = bron['id']

# find team Ids
from nba_api.stats.static import teams 
teams = teams.get_teams()
GSW = [x for x in teams if x['full_name'] == 'Golden State Warriors'][0]
GSW_id = GSW['id']

#Call the API endpoint passing in lebron's ID & which season 
gamelog_bron = playergamelog.PlayerGameLog(player_id='2544', season = '2018')

#Converts gamelog object into a pandas dataframe
#can also convert to JSON or dictionary  
df_bron_games_2018 = gamelog_bron.get_data_frames()

# If you want all seasons, you must import the SeasonAll parameter 
from nba_api.stats.library.parameters import SeasonAll

gamelog_bron_all = playergamelog.PlayerGameLog(player_id='2544', season = SeasonAll.all)

df_bron_games_all = gamelog_bron_all.get_data_frames()

from nba_api.stats.endpoints import leaguegamefinder

#this time we convert it to a dataframe in the same line of code
GSW_games = leaguegamefinder.LeagueGameFinder(team_id_nullable=GSW_id).get_data_frames()[0]

###### playing around with this ################

def season_string(year):
    return f"{year}-{str(year+1)[-2:]}"

current_year = 2024  # update if needed
seasons = [season_string(y) for y in range(current_year-5, current_year)]

# Warriors team ID
WARRIORS_ID = 1610612744

dfs = []

for season in seasons:
    logs = TeamGameLogs(team_id_nullable=WARRIORS_ID, season_nullable=season)
    df = logs.get_data_frames()[0]
    df["SEASON"] = season
    dfs.append(df)

games = pd.concat(dfs, ignore_index=True)

# -------------------------
# 2. Clean / convert fields
# -------------------------

games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
games = games.sort_values("GAME_DATE")

# Points scored by Warriors in each game:
games["PTS"] = games["PTS"].astype(int)

# -------------------------
# 3. Plot using seaborn / matplotlib
# -------------------------

plt.figure(figsize=(14,6))
sns.lineplot(data=games, x="GAME_DATE", y="PTS", marker="o", linewidth=1)

plt.title("Golden State Warriors — Points Scored Per Game (Last 5 Seasons)")
plt.xlabel("Game Date")
plt.ylabel("Points Scored")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()