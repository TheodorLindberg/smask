import numpy as np
import pandas as pd
import sklearn.linear_model as sk_lm
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
import sklearn.preprocessing as skl_pre
import sklearn.discriminant_analysis as skl_da
training_data = '/Users/alfredaxelsson/Desktop/python/project smask/training_data_VT2026.csv'

data = pd.read_csv(training_data)

np.random.seed(1)

trainI = np.random.choice(data.shape[0],size =1000, replace=False)
trainIndex = data.index.isin(trainI)
train = data.iloc[trainIndex]
test = data.iloc[~trainIndex]


training_var = ['hour_of_day','day_of_week','month','holiday','weekday','summertime','temp','dew','humidity','precip','snow','snowdepth','windspeed','cloudcover','visibility']
var_test = {}
print('KNN')
for i in training_var:
    x_a_train = train[[i]]
    y_a_train = train['increase_stock']
    x_a_test = test[[i]]
    y_a_test = test['increase_stock']
    model = skl_nb.KNeighborsClassifier(n_neighbors=11)

    model.fit(x_a_train,y_a_train)

    prediction = model.predict(x_a_test)
    print(i,np.mean(prediction == y_a_test))

x_train = train[['hour_of_day','day_of_week','month','holiday','weekday','summertime','temp','dew','humidity','precip','snow','snowdepth','windspeed','cloudcover','visibility']]
y_train = train['increase_stock']
x_test = test[['hour_of_day','day_of_week','month','holiday','weekday','summertime','temp','dew','humidity','precip','snow','snowdepth','windspeed','cloudcover','visibility']]
y_test = test['increase_stock']
scaler = skl_pre.StandardScaler().fit(x_train)
missclass = []
for k in range(1,51):

    model = skl_nb.KNeighborsClassifier(n_neighbors=k)

    model.fit(scaler.transform(x_train),y_train)

    prediction = model.predict(scaler.transform(x_test))
    missclass.append(np.mean(prediction != y_test))
K = np.linspace(1,50,50)
plt.plot(K,missclass,'.')

missclass_non = []
for k in range(1,51):

    model = skl_nb.KNeighborsClassifier(n_neighbors=k)

    model.fit(x_train,y_train)

    prediction = model.predict(x_test)
    missclass_non.append(np.mean(prediction != y_test))
K = np.linspace(1,50,50)
plt.plot(K,missclass_non,'*')
plt.show()
#print(pd.crosstab(prediction,y_test))
#print(f'accuracy:{np.mean(prediction == y_test):.3f}')
print(var_test)