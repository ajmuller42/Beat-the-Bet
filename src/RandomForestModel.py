import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from nba_api.stats.endpoints import LeagueGameLog
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import joblib
import os

label_encoder = LabelEncoder()

def get_season_data(season):
    log = LeagueGameLog(season=season)
    df = log.get_data_frames()[0]
    df['SEASON'] = season
    return df

seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2018, 2024)]
df = pd.concat([get_season_data(s) for s in seasons], ignore_index=True)

df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

df['HOME'] = df['MATCHUP'].str.contains(' vs. ').astype(int)

home_df = df[df['HOME'] == 1].copy()
away_df = df[df['HOME'] == 0].copy()

games = home_df.merge(
    away_df,
    on='GAME_ID',
    suffixes=('_HOME', '_AWAY')
)

games['HOME_WIN'] = (games['PTS_HOME'] > games['PTS_AWAY']).astype(int)

features = [
    'PTS_HOME', 'REB_HOME', 'AST_HOME', 'STL_HOME', 'BLK_HOME', 'TOV_HOME',
    'PTS_AWAY', 'REB_AWAY', 'AST_AWAY', 'STL_AWAY', 'BLK_AWAY', 'TOV_AWAY'
]

X = games[features]
y = games['HOME_WIN']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

scores = cross_val_score(
    model,
    X,
    y,
    cv = 5,
    scoring = 'accuracy'
)

print("Model accuracy:", accuracy)
print("F1 Score:", f1)
print("CV accuracy scores:", scores)
print("Mean CV accuracy:", scores.mean())


df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

for stat in ['PTS', 'REB', 'AST', 'TOV']:
    df[f'{stat}_ROLL5'] = df.groupby('TEAM_ID')[stat].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, 'RandomForestModel.pkl')
joblib.dump(model, model_path)