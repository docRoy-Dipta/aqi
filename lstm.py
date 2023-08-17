# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 03:08:29 2023

@author: roy
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import tensorflow as tf
from tensorflow.keras import regularizers


# Load the data
data = pd.read_csv(r"C:\Users\User\Documents\Dhaka Air Quality\dhaka-air-quality.csv")
data.head()



# Drop AQI values less than 0
data.drop(data[data['AQI'] < 0].index, inplace = True)

data.rename(columns = {"NowCast Conc.":'NowCast'}, inplace = True)
data.rename(columns = {"Raw Conc.":'RawConc'}, inplace = True)

data.drop(['Hour','AQI Category','QC Name','Unnamed: 8','Conc. Unit'], axis=1, inplace=True)
data.head()

data.info


# LSTM

# Prepare the data
x = data[["NowCast", "RawConc"]]
y = data["AQI"]

# Scale the data
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

with open(r'C:\Users\User\Documents\Capstone\models\scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Reshape the input tensor
x_reshaped = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, activation="tanh", return_sequences=True, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001)))
model.add(LSTM(64, activation="tanh", kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001)))
model.add(Dense(1, activation="linear"))

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

# Perform K-Fold Cross-Validation
n_splits = 5  # Number of folds for cross-validation
kf = KFold(n_splits=n_splits, shuffle=True)

# Initialize lists to store error metrics for each fold
mae_train_list = []
mae_test_list = []
mape_train_list = []
mape_test_list = []
rmse_train_list = []
rmse_test_list = []
epsilon = 1e-8  # A small epsilon value to avoid division by zero


for train_index, test_index in kf.split(x_reshaped):
    x_train, x_test = x_reshaped[train_index], x_reshaped[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(x_train, y_train, epochs=40, verbose=1)

    # Make predictions on train and test sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Reshape the predictions to be 1-dimensional
    y_train_pred_inv = y_train_pred.reshape(-1, 1)
    y_test_pred_inv = y_test_pred.reshape(-1, 1)
    y_train_true_inv = y_train.values.reshape(-1, 1)
    y_test_true_inv = y_test.values.reshape(-1, 1)

    # Calculate MAE, MAPE, and RMSE for train and test sets
    mae_train = mean_absolute_error(y_train_true_inv, y_train_pred_inv)
    mae_test = mean_absolute_error(y_test_true_inv, y_test_pred_inv)

    mape_train = np.mean(np.abs((y_train_true_inv - y_train_pred_inv) / (y_train_true_inv + epsilon))) * 100
    mape_test = np.mean(np.abs((y_test_true_inv - y_test_pred_inv) / (y_test_true_inv + epsilon))) * 100

    rmse_train = np.sqrt(mean_squared_error(y_train_true_inv, y_train_pred_inv))
    rmse_test = np.sqrt(mean_squared_error(y_test_true_inv, y_test_pred_inv))

    # Append error metrics to the lists
    mae_train_list.append(mae_train)
    mae_test_list.append(mae_test)
    mape_train_list.append(mape_train)
    mape_test_list.append(mape_test)
    rmse_train_list.append(rmse_train)
    rmse_test_list.append(rmse_test)
# Calculate the mean error metrics for train and test sets
mean_mae_train = np.mean(mae_train_list)
mean_mae_test = np.mean(mae_test_list)

mean_mape_train = np.mean(mape_train_list)
mean_mape_test = np.mean(mape_test_list)

mean_rmse_train = np.mean(rmse_train_list)
mean_rmse_test = np.mean(rmse_test_list)

# Print the mean error metrics for train and test sets
print("\nMean Absolute Error (MAE) Train:", mean_mae_train)
print("Mean Absolute Error (MAE) Test:", mean_mae_test)

print("\nMean Absolute Percentage Error (MAPE) Train:", mean_mape_train)
print("Mean Absolute Percentage Error (MAPE) Test:", mean_mape_test)

print("\nRoot Mean Squared Error (RMSE) Train:", mean_rmse_train)
print("Root Mean Squared Error (RMSE) Test:", mean_rmse_test)


# Evaluate the model
model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)

# Reshape the predictions and y_test variables
predictions = predictions.reshape(-1)

# Calculate the differences between predicted and actual values
diff = predictions - y_test

# Print the differences
print(diff)


# Convert y_test Series to a NumPy array and reshape
real_values = y_test.values.reshape(-1)
predicted_values = predictions.reshape(-1)

# Create a dataframe with the real and predicted values
df = pd.DataFrame({"real": real_values, "predicted": predicted_values})

# Print the dataframe
print(df)


model.save(r'C:\Users\User\Documents\Capstone\models\lstm_model.h5')



loaded_model = tf.keras.models.load_model(r'C:\Users\User\Documents\Capstone\models\lstm_model.h5')


# Prepare input data for prediction
specific_date = pd.to_datetime('2023-06-20')
specific_time = '01:00:00'

# Create a DataFrame with the specific date and time
input_data = pd.DataFrame({'date': [specific_date], 'time': [specific_time]})
# Add other features like NowCast and RawConc for the specific date and time
input_data['NowCast'] = 60.5  # Replace with your NowCast data for the specific date and time
input_data['RawConc'] = 59  # Replace with your RawConc data for the specific date and time

# Scale the input data
input_data_scaled = scaler.transform(input_data[['NowCast', 'RawConc']])

# Reshape the input tensor
input_data_reshaped = input_data_scaled.reshape(1, 2, 1)

# Make predictions using the Bi-GRU model
predicted_aqi = loaded_model.predict(input_data_reshaped)

# Inverse transform the predicted AQI value to the actual scale
predicted_data = pd.DataFrame({
    'NowCast': [input_data_scaled[0, 0]],
    'RawConc': [input_data_scaled[0, 1]],
    'Predicted_AQI': [predicted_aqi[0, 0]]
})
# Inverse transform the 'NowCast' and 'RawConc' columns to the actual scale
predicted_data_actual_scale = predicted_data.copy()
predicted_data_actual_scale[['NowCast', 'RawConc']] = scaler.inverse_transform(predicted_data[['NowCast', 'RawConc']])

# Extract the predicted AQI value from the DataFrame
predicted_aqi_actual_scale = predicted_data_actual_scale['Predicted_AQI'].values[0]

print(f"Predicted AQI for {specific_date} at {specific_time}: {predicted_aqi_actual_scale}")


