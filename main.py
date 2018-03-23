import requests
import numpy as np

"""
  get data from api
"""
api_key = ""

url = "https://www.quandl.com/api/v3/datasets/WIKI/FB/data.json?&api_key={}".format(api_key)
res = requests.get(url)
data = res.json()['dataset_data']['data']

"""
  extract features and labels

  features: all the fields in the api
  labels  : `high` in the previous day(will be the last row)
"""

features = []
labels   = []

for i in range(0, len(data) - 1):
  features.append(data[i + 1][1:])
  labels.append(data[i][4:5]) # 5 is the index of the closing price


# convert to numpy array
features = np.asarray(features)
labels = np.asarray(labels)

"""
  split data
"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
  features, labels, test_size=0.05, random_state=42)

"""
  rescale data
"""
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(x_train)

x_train_transformed = scaler.transform(x_train)

"""
  train dataset using lasso regression
"""
from sklearn.svm import SVR

svr_rbf = SVR(kernel='linear')
svr_rbf.fit(x_train_transformed, y_train)

"""
  score testing set
"""
from sklearn.metrics import mean_squared_error, r2_score

# make prediction
y_pred = svr_rbf.predict(scaler.transform(x_test))

# The coefficients
# print('Coefficients: \n', svr_rbf.coef_)

# The mean squared error
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))

# Explained variance score: 1 is perfect prediction
print('Variance score: {}'.format(r2_score(y_test, y_pred)))

# pridict the high for tomorrow
# print(reg.predict(scaler.transform([data[0][1:], data[1][1:]])))
print(svr_rbf.predict(scaler.transform([data[0][1:], data[1][1:]])))

