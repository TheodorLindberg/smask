
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.dummy import DummyClassifier
RANDOM_STATE = 42

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def load_data():
        
    df = pd.read_csv("training_data.csv")

    # Map labels to integers if needed
    df['increase_stock_label'] = df['increase_stock'].map({
        "low_bike_demand": 0,
        "high_bike_demand": 1
    })

    y = df['increase_stock_label'] 

    X = df.drop(columns=["increase_stock", "increase_stock_label"])

    return X, y


def pre_process_training_data_cylic(X):
    X["month"] = X["month"] -1

    X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)

    X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    X = X.drop(columns=["hour_of_day", "day_of_week", "month"])

    return X


def pre_process_training_data(X):
    
    return X

def dummy_classifier(X_train, y_train, X_test):
    classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(X_train, y_train)

    return classifier.predict(X_test)

    