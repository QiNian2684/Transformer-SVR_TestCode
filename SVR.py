import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Parameters
FLOOR_HEIGHT = 6  # Floor height in meters
SVR_PARAMS = {'C': 1.0, 'epsilon': 0.1, 'kernel': 'rbf'}  # Default SVR parameters

# Load data
train_data = pd.read_csv('UJIndoorLoc/trainingData_building0.csv')
validation_data = pd.read_csv('UJIndoorLoc/validationData_building0.csv')

# Select data for building 0
train_data = train_data[train_data['BUILDINGID'] == 0]
validation_data = validation_data[validation_data['BUILDINGID'] == 0]

# Separate features and targets
features_train = train_data.iloc[:, :-9]  # Assuming the last 9 columns are target and location info
target_train = train_data.loc[:, ['LONGITUDE', 'LATITUDE', 'FLOOR']]

features_val = validation_data.iloc[:, :-9]
target_val = validation_data.loc[:, ['LONGITUDE', 'LATITUDE', 'FLOOR']]

# Train models
svr_lon = SVR(**SVR_PARAMS)
svr_lat = SVR(**SVR_PARAMS)
svr_floor = SVR(**SVR_PARAMS)

svr_lon.fit(features_train, target_train['LONGITUDE'])
svr_lat.fit(features_train, target_train['LATITUDE'])
svr_floor.fit(features_train, target_train['FLOOR'])

# Predict
pred_lon = svr_lon.predict(features_val)
pred_lat = svr_lat.predict(features_val)
pred_floor = svr_floor.predict(features_val) * FLOOR_HEIGHT  # Convert floors to meters

# Calculate MSE
mse_lon = mean_squared_error(target_val['LONGITUDE'], pred_lon)
mse_lat = mean_squared_error(target_val['LATITUDE'], pred_lat)
mse_floor = mean_squared_error(target_val['FLOOR'] * FLOOR_HEIGHT, pred_floor)  # Convert to meters

# Predicted and actual coordinates
predictions = np.vstack((pred_lon, pred_lat, pred_floor)).T
actuals = np.vstack((target_val['LONGITUDE'], target_val['LATITUDE'], target_val['FLOOR'] * FLOOR_HEIGHT)).T

# Calculate Euclidean distance
differences = np.sqrt(np.sum((predictions - actuals) ** 2, axis=1))

# Calculate average 3D error
average_3d_error = np.mean(differences)

print("Predicted coordinates (meters):", list(zip(pred_lon, pred_lat, pred_floor)))
print("MSE for Longitude: {:.2f}, Latitude: {:.2f}, Floor Height: {:.2f} meters".format(mse_lon, mse_lat, mse_floor))
print("Average 3D error (meters):", average_3d_error)
