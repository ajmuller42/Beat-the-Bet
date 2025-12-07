import pandas as pd 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib

def trainModel():
    data = pd.read_csv('dataset.csv')
    data = data.dropna(subset = ['result'])

    x = data[[
        'offensiveRatingDifference',
        'defensiveRatingDifference',
        'totalScoreDifference',
        'assistDifference',
        'turnoverDifference'
    ]]

    y = data['result']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = False)
    model = XGBClassifier(n_estimators = 300, learning_rate = 0.03, max_depth = 5)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Accuracy: ", accuracy_score(y_test, predictions))
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    joblib.dump(model, model_path)

    print("The Model has been successfully saved to model.pkl!")

if __name__ == "__main__":
    trainModel()

    