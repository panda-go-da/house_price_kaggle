'''
Position 35802 out of 72507, with result 17600.84102
Position 25084 out of 72507, with result 16597.33262 fillna(0)
16648.42969 fillna(method='ffill')
16621.63126 fillna(method='bfill')
'''

# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

# Path of the file to read
training_data = 'train.csv'
home_data = pd.read_csv(training_data)

home_data = home_data.fillna(0)

# set target of prediction
y = home_data.SalePrice

# set features as base of prediction
features = ['MSSubClass',
'LotArea',
'OverallQual',
'OverallCond',
'YearBuilt',
'YearRemodAdd',
'1stFlrSF',
'2ndFlrSF',
'LowQualFinSF',
'GrLivArea',
'FullBath',
'HalfBath',
'BedroomAbvGr',
'KitchenAbvGr',
'TotRmsAbvGrd',
'Fireplaces',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal',
'MoSold',
'YrSold']

# Select columns corresponding to features, and preview the data
# X = home_data[features]
X = home_data.select_dtypes(include=['int64', 'float64'])
print(X.info())
X = X.drop('SalePrice', axis=1)
print(X.info())

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# Validation MAE for Random Forest Model: 17,768

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions
test_data_path = 'test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
# test_X = test_data[features]
test_X = test_data.select_dtypes(include=['int64', 'float64']).fillna(0)

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)


