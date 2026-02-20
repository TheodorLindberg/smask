import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
)

import helper;

def gradiant_boost(X_train, y_train):
    param_grid_gb = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [1.0]
    }

    gb_gs = GridSearchCV(
        GradientBoostingClassifier(random_state=helper.RANDOM_STATE),
        param_grid_gb,
        cv=CV,
        scoring='accuracy',
        n_jobs=-1
    )

    gb_gs.fit(X_train, y_train)

    print(f'Best params: {gb_gs.best_params_}')
    print(f'Best CV accuracy: {gb_gs.best_score_:.4f}')

    return gb_gs
