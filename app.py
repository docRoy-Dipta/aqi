# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 04:07:45 2023

@author: User
"""

# https://youtu.be/bluclMxiUkA
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 

Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.

"""


from flask import Flask, request, render_template
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model.
model = tf.keras.models.load_model(r'C:\Users\User\Documents\Capstone\models\lstm_model.h5')

# Load the scaler
with open(r'C:\Users\User\Documents\Capstone\models\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()] # Convert string inputs to float.
    features = np.array(int_features)  # Convert to a numpy array

    nowcast = features[0]  # Fetch 'NowCast' value from HTML form
    rawconc = features[1]  # Fetch 'RawConc' value from HTML form

    input_data = pd.DataFrame({
        'NowCast': [nowcast],
        'RawConc': [rawconc]
    })

    input_data_scaled = scaler.transform(input_data[['NowCast', 'RawConc']])  # Use the loaded scaler
    input_data_reshaped = input_data_scaled.reshape(1, 2, 1)

    predicted_aqi = model.predict(input_data_reshaped)  # Use the loaded model

    predicted_data = pd.DataFrame({
        'NowCast': [input_data_scaled[0, 0]],
        'RawConc': [input_data_scaled[0, 1]],
        'Predicted_AQI': [predicted_aqi[0, 0]]
    })
    predicted_data_actual_scale = predicted_data.copy()
    predicted_data_actual_scale[['NowCast', 'RawConc']] = scaler.inverse_transform(predicted_data[['NowCast', 'RawConc']])
    predicted_aqi_actual_scale = predicted_data_actual_scale['Predicted_AQI'].values[0]

    return render_template('index.html', prediction_text=f"Predicted AQI: {predicted_aqi_actual_scale:.2f}")



#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()
