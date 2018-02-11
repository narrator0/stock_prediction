import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""
  get data from api
"""
url = 'https://www.quandl.com/api/v3/datasets/WIKI/FB/data.json'

res = requests.get(url)

data = res.json()['dataset_data']['data']

# data = data[0:10]

"""
  extract features and labels

  features: all the fields in the api
  labels  : `high` in the previous day(will be the last row)
"""
features = []
labels   = []

for i in range(0, len(data) - 1):
  features.append(data[i][1:])
  labels.append(data[i + 1][2:3]) # the index of high from api is 2
 
# convert to numpy array
features = np.asarray(features)
labels   = np.asarray(labels)

"""
  spit data
"""

X_train, X_test, y_train, y_test = train_test_split(
  features, labels, test_size=0.33, random_state=42)

"""
  train dataset using lasso regression
"""

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

"""
  score testing set
"""

# make prediction
y_pred = reg.predict(X_test)

# The coefficients
print('Coefficients: \n', reg.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# pridict the high for tomorrow
print(reg.predict([data[0][1:]]))
