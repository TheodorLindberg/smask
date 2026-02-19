
import pandas as pd

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


def pre_process_training_data(X):
    X["month"] = X["month"] -1


def dummy_classifier():
    