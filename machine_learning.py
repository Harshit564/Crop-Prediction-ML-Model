import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import joblib

# from krishisathi.settings import BASE_DIR


def train_model():
    data = pd.read_csv('final_crops_data.csv')
    X = data.iloc[:, 0:4]
    y = data.iloc[:, -1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # clf = RandomForestClassifier(n_estimators=13)
    clf = ExtraTreesClassifier(n_estimators=200, random_state=7)
    clf.fit(X, y)

    # Save the trained model to the disk
    filename = 'trained_model.sav'
    joblib.dump(clf, filename)


def test_model(new_data):
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))), 'machine_learning/trained_model.sav')
    model = joblib.load(model_path)
    # X_new = [[5.1, 0.5, 0.5, 0.3]]
    y_prediction = model.predict([new_data])
    print(y_prediction)
    return y_prediction


Features = [5.1, 0.5, 0.5, 0.3]

if __name__ == '__main__': test_model(Features)
