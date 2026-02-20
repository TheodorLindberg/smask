import numpy as np
import pandas as pd
import sklearn.linear_model as sk_lm
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
import sklearn.preprocessing as skl_pre

training_data = 'training_data_VT2026.csv'

data = pd.read_csv(training_data)

np.random.seed(1)

trainI = np.random.choice(data.shape[0],size =1000, replace=False)
trainIndex = data.index.isin(trainI)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]

training_var = ['hour_of_day','day_of_week','month','holiday','weekday','summertime','temp','dew','humidity','precip','snow','snowdepth','windspeed','cloudcover','visibility']
x_train = train[training_var]
y_train = train['increase_stock']
x_test = test[training_var]
y_test = test['increase_stock']
scaler = skl_pre.StandardScaler().fit(x_train)
model = skl_nb.KNeighborsClassifier(n_neighbors=11)
model.fit(scaler.transform(x_train),y_train)
prediction = model.predict(scaler.transform(x_test))
print(pd.crosstab(prediction,y_test))
print(f'accuracy:{np.mean(prediction == y_test):.3f}')