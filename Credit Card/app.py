from flask import Flask, request, redirect, url_for, send_file, render_template
import pandas as pd
import joblib
import numpy as np
from geopy.distance import great_circle
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calculate_distance(row):
    return great_circle(
        (row['customer_latitude'], row['customer_longitude']),
        (row['merchant_latitude'], row['merchant_longitude'])
    ).km

def get_unusual_rating(hour):
    if 22 <= hour < 24 or 0 <= hour < 8:
        return 1
    return 0

def get_distance_rating(distance):
    if distance > 15000:
        return 1
    elif distance > 10000:
        return 0.75
    elif distance > 5000:
        return 0.5
    else:
        return 0.25

def get_frequency_rating(row, frequency):
    return min(1, frequency[(frequency['customer_latitude'] == row['customer_latitude']) & 
                             (frequency['customer_longitude'] == row['customer_longitude']) & 
                             (frequency['transaction_date'] == row['transaction_date'])]['frequency'].values[0] / 5)

def get_fraud_risk(average_rating):
    if 0.0 <= average_rating < 0.4:
        return 'low'
    elif 0.4 <= average_rating < 0.7:
        return 'moderate'
    else:
        return 'high'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the data
        df = pd.read_excel(filepath)

        # Add new columns
        df['unusual_rating'] = df['transaction_hour'].apply(get_unusual_rating)
        df['distance'] = df.apply(calculate_distance, axis=1)
        df['distance_rating'] = df['distance'].apply(get_distance_rating)
        df['state_rating'] = (df['customer_state'] != df['merchant_state']).astype(int)

        # Calculate transaction frequency
        frequency = df.groupby(['customer_latitude', 'customer_longitude', 'transaction_date']).size().reset_index(name='frequency')
        df['limit_rating'] = df.apply(lambda row: get_frequency_rating(row, frequency), axis=1)

        # Calculate average rating
        df['average_rating'] = (df['distance_rating'] + df['state_rating'] + df['limit_rating'] + df['unusual_rating']) / 4

        # Predict fraud risk
        features = ['distance_rating', 'state_rating', 'limit_rating', 'unusual_rating']
        X = df[features]
        X_scaled = scaler.transform(X)  # Use the fitted scaler for transformation
        df['fraud_risk'] = model.predict(X_scaled)
        
        # Save the processed file
        processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + filename)
        df.to_excel(processed_file_path, index=False)

        return send_file(processed_file_path, as_attachment=True)

    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    app.run(debug=True)
