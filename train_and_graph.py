import pprint
import pickle
import requests
import numpy as np

pp = pprint.PrettyPrinter(indent=2)

"""
  get data from api
"""
url = 'https://www.quandl.com/api/v3/datasets/WIKI/FB/data.json'

res = requests.get(url)

data = np.asarray(res.json()['dataset_data']['data'])

"""
  read data from data.pkl
"""
# file = open('data.pkl')
# original_data = pickle.load(file)
# data = np.asarray(original_data['dataset_data']['data'])

# file.close()

"""
  extract date and data.
"""
dates = data[:, 0].astype('datetime64')
data  = data[:, 1:].astype('float32')

# extract data before 2018.02.01
cut_point = 0

for i in range(0, len(dates)):
  if (dates[i] == np.datetime64('2018-02-01')):
    cut_point = i + 1
    break

dates_train = dates[cut_point:]
data_train  = data[cut_point:]

"""
  extract feature and train datasets
"""
# the feature is the data of the previous day
# so we have to exclude the last day because
# there won't be a feature for the last day
features = data_train[1:, :]
labels   = np.delete(data_train[:, 1], -1)

"""
  spit data
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  features, labels, test_size=0.33, random_state=42)

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

reg = linear_model.LinearRegression()
reg.fit(X_train_transformed, y_train)

"""
  score testing set
"""
from sklearn.metrics import mean_squared_error, r2_score

# make prediction
y_pred = reg.predict(scaler.transform(X_test))

# The coefficients
print("Coefficients: \n", reg.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print("Variance score: %.2f" % r2_score(y_test, y_pred))

"""
  plot actual high and predict high from 2018-02-01 to 2018-02-28
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots()

predictions = reg.predict(scaler.transform(data[:cut_point]))

ax.plot(
  dates[:cut_point].astype('O'), data[:cut_point, 1], 'b-',
  dates[:cut_point].astype('O'), predictions, 'r-')

blue_patch = mpatches.Patch(color='blue', label='actual')
red_patch = mpatches.Patch(color='red', label='predict')
plt.legend(handles=[blue_patch, red_patch])

# format tricker
import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

ax.yaxis.set_major_locator(plt.LinearLocator())

plt.show()



