import requests
import numpy as np

"""
  get data from api
"""
url = 'https://www.quandl.com/api/v3/datasets/WIKI/FB/data.json'
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
  labels.append(data[i][4:5]) # 4 is the index of the closing price
  #labels.append(data[i][2:3]) # the index of high from api is 2

# convert to numpy array
features = np.asarray(features)
labels   = np.asarray(labels)

"""
  split data
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  features, labels, test_size=0.05, random_state=42)

"""
  rescale data
"""
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_transformed = scaler.transform(X_train)

"""
  train dataset using lasso regression
"""
from sklearn import linear_model
from sklearn.svm import SVR

# reg = linear_model.LinearRegression()
# reg.fit(X_train_transformed, y_train)

svr_rbf = SVR(kernel='linear', C=1e3)
svr_rbf.fit(X_train_transformed, y_train)

"""
  score testing set
"""
from sklearn.metrics import mean_squared_error, r2_score

# make prediction
y_pred = svr_rbf.predict(scaler.transform(X_test))

# The coefficients
# print('Coefficients: \n', svr_rbf.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# pridict the high for tomorrow
# print(reg.predict(scaler.transform([data[0][1:], data[1][1:]])))
print(svr_rbf.predict(scaler.transform([data[0][1:], data[1][1:]])))

