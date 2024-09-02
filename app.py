from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pyowm
from pyowm import OWM
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.secret_key = 'API_KEY'  # Required for using sessions and flash messages

# Load environment variables
load_dotenv()
api_key = os.getenv('API_KEY')

# Dummy credentials for simplicity
USERNAME = 'admin'
PASSWORD = '12345'

# Function to fetch weather data
def get_weather_data(api_key, location):
    try:
        owm = OWM(api_key)
        mgr = owm.weather_manager()
        observation = mgr.weather_at_place(location)
        weather = observation.weather
        return {
            'main': {
                'temp': weather.temperature('celsius')['temp']
            }
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

# Route for the login page
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials, please try again.", "danger")
            return redirect(url_for('login'))

    return render_template("login.html")

# Route for the home page
@app.route("/home")
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template("home.html")

# Route for the about page
@app.route('/about')
def about():
    return render_template('aboutus.html')

# Route to fetch available cities and their weather data
@app.route('/get-temperature', methods=['GET'])
def get_temperature():
    city = request.args.get('city')

    if city:
        location = f'{city},IN'  # Assuming the country code for India is 'IN'
        weather_data = get_weather_data(api_key, location)
        if weather_data:
            temperature = weather_data['main']['temp']
            return jsonify({'temperature': temperature})
        else:
            return jsonify({'error': 'Failed to fetch weather data'}), 500
    else:
        return jsonify({'error': 'City not provided'}), 400

# Route for the prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == "POST":
        try:
            # Get form data
            water = float(request.form["water"])
            UV = float(request.form["UV"])
            area = float(request.form["area"])
            fertilizer = float(request.form["fertilizer"])
            Pesticide = float(request.form["Pesticide"])
            Region = float(request.form["Region"])

            # Prepare data for prediction
            sample_data = [water, UV, area, fertilizer, Pesticide, Region]
            ex1 = np.array(sample_data).reshape(1, -1)

            # Load dataset
            data = pd.read_csv("dataset.csv")
            data = data.drop(columns=['id', 'categories'], axis=1)

            # Replace NaN values with median
            data.water = data.water.fillna(data.water.median())
            data.uv = data.uv.fillna(data.uv.median())

            # Remove outliers
            data = data[data['water'] <= 200]

            # Split data into training and test sets
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

            # Train the model
            regressor = RandomForestRegressor(n_estimators=10, random_state=50)
            regressor.fit(X_train, y_train)

            # Make a prediction
            yhat = regressor.predict(ex1)
            res = yhat[0]
            area = int(area)
            result = 27.6 * res * area

            return render_template('CropResult.html', prediction_text=res, area=area, result=result)
        except Exception as e:
            flash(f"An error occurred: {e}", "danger")
            return redirect(url_for('home'))

# Route for the logout page
@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
