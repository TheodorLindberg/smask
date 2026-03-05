from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

import helper
X, y = helper.load_data()   

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=helper.RANDOM_STATE, stratify=y
)

X_train = helper.pre_process_training_data(X_train_raw)
X_test = helper.pre_process_training_data(X_test_raw)



# Pipeline
pipe = Pipeline([
    ('bagging', BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=1))
])

# Grid
param_grid = {
    'bagging__n_estimators': [10, 20, 50, 100],
    'bagging__max_samples': [0.5, 0.8, 1.0],
    'bagging__estimator__max_depth': [3, 5, None],
    'bagging__estimator__min_samples_split': [2, 5, 10]
}

# Grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)

best_model = grid.best_estimator_
y_predict = best_model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_predict))
print(classification_report(y_test, y_predict))


# Test data prediction
x_final = pd.read_csv("test_data_VT2026.csv")
y_predict_final = best_model.predict(x_final)
np.savetxt("output.csv", [y_predict_final], delimiter=",", fmt="%d")
print(y_predict_final)