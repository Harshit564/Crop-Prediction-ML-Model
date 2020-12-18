import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model():
    data = pd.read_csv('final_crops_data.csv')
    X = data.iloc[:, 0:4]
    y = data.iloc[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # clf = RandomForestClassifier(n_estimators=13)

    # Hyperparameter Tuning
    #  from sklearn.model_selection import GridSearchCV
    # # Create the parameter grid based on the results of random search
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [10,20,30,40,50],
    #     'max_features': [2, 3],
    #     'min_samples_leaf': [3, 4, 5],
    #     'min_samples_split': [8, 10, 12],
    #     'n_estimators': [100, 200, 300, 1000]
    # }
    # # Create a based model
    # rf_clf_grid = RandomForestClassifier()
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator=rf_clf_grid, param_grid=param_grid,
    #                         cv=3, n_jobs=-1, verbose=2)

    # grid_search.fit(X_train_std_scaled, y_train)
    # grid_search.best_params_
    # {'bootstrap': True,
    #  'max_depth': 80,
    #  'max_features': 2,
    #  'min_samples_leaf': 5,
    #  'min_samples_split': 8,
    #  'n_estimators': 100}

    clf = RandomForestClassifier(n_estimators=50,
                                 min_samples_split=8,
                                 min_samples_leaf=5,
                                 max_features=4,
                                 max_depth=10,
                                 bootstrap=True,
                                 random_state=50)
    clf.fit(X, y)

    # Save the trained model to the disk
    filename = 'trained_model.sav'
    joblib.dump(clf, filename)


def test_model(new_data):
    model_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath('__file__'))), 'machine_learning/trained_model.sav')
    model = joblib.load(model_path)
    # X_new = [[5.1, 0.5, 0.5, 0.3]]
    y_prediction = model.predict([new_data])
    print(y_prediction)
    return y_prediction


Features = [5.1, 0.5, 0.5, 0.3]

if __name__ == '__main__':
    test_model(Features)
