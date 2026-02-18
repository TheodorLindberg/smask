
import pandas as pd
df = pd.read_csv("training_data.csv")

# Map labels to integers if needed
df['increase_stock_label'] = df['increase_stock'].map({
    "low_bike_demand": 0,
    "high_bike_demand": 1
})

df["month"] = df["month"] -1

y = df['increase_stock_label'] 
X = df.drop(columns=["increase_stock", "increase_stock_label"])

import numpy as np

np.random.seed(1)
N = len(X)
M = np.ceil(0.8 * N).astype(int)  # Number of training data

idx = np.random.permutation(N)

X_train, X_test = X.iloc[idx[:M]], X.iloc[idx[M:]]
y_train, y_test = y.iloc[idx[:M]], y.iloc[idx[M:]]



