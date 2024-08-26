from flask import Flask, request, send_file, render_template_string
import pandas as pd
import io
import pickle
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.simplefilter('ignore', InconsistentVersionWarning)

app = Flask(__name__)

# Path to your model
MODEL_PATH = 'model.pkl'

# Load the model
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

# Define the feature columns for model prediction
FEATURE_COLUMNS = [
    'unusual_rating', 'distance_rating', 'state_rating', 'limit_rating', 'average_rating'
]

def preprocess_data(df):
    """
    Manually preprocess data including feature scaling.
    """
    columns_to_standardize = [
        'unusual_rating',
        'distance_rating',
        'state_rating',
        'limit_rating',
        'average_rating'
    ]
    
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    
    return df

def process_dataframe(df):
    """
    Process the DataFrame by converting data types, handling missing values,
    and deriving new feature columns.
    """
    # Handle missing values for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Convert categorical columns to numeric codes if necessary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes
    
    def get_unusual_rating(hour):
        if 22 <= hour < 24 or 0 <= hour < 8:
            return 1
        return 0

    if 'transaction_hour' in df.columns:
        df['unusual_rating'] = df['transaction_hour'].apply(get_unusual_rating)
    else:
        df['unusual_rating'] = 0

    def calculate_distance(row):
        return great_circle(
            (row['customer_latitude'], row['customer_longitude']),
            (row['merchant_latitude'], row['merchant_longitude'])
        ).km

    if all(col in df.columns for col in ['customer_latitude', 'customer_longitude', 'merchant_latitude', 'merchant_longitude']):
        df['distance'] = df.apply(calculate_distance, axis=1)
    else:
        df['distance'] = 0

    def get_distance_rating(distance):
        if distance > 15000:
            return 1
        elif distance > 10000:
            return 0.75
        elif distance > 5000:
            return 0.5
        else:
            return 0.25

    df['distance_rating'] = df['distance'].apply(get_distance_rating)
    
    if 'customer_state' in df.columns and 'merchant_state' in df.columns:
        df['state_rating'] = (df['customer_state'] != df['merchant_state']).astype(int)
    else:
        df['state_rating'] = 0

    df['limit_rating'] = 0  # Placeholder

    if 'customer_latitude' in df.columns and 'customer_longitude' in df.columns and 'transaction_date' in df.columns:
        frequency = df.groupby(['customer_latitude', 'customer_longitude', 'transaction_date']).size()
        frequency = frequency.reset_index(name='frequency')

        def get_frequency_rating(row):
            freq = frequency[(frequency['customer_latitude'] == row['customer_latitude']) & 
                             (frequency['customer_longitude'] == row['customer_longitude']) & 
                             (frequency['transaction_date'] == row['transaction_date'])]['frequency']
            return min(1, freq.values[0] / 5) if not freq.empty else 0

        df['limit_rating'] = df.apply(get_frequency_rating, axis=1)
    
    df['average_rating'] = (df['distance_rating'] + df['state_rating'] + df['limit_rating'] + df['unusual_rating']) / 4
    
    df = preprocess_data(df)

    return df

@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Upload Test</title>
        </head>
        <body>
            <h1>Upload Test</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file1">
                <button type="submit">Upload</button>
            </form>
        </body>
        </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    if 'file1' not in request.files:
        return 'Missing file(s)', 400

    file1 = request.files['file1']
    
    if file1.filename == '':
        return 'No selected file', 400

    try:
        # Load Excel file into DataFrame
        df = pd.read_excel(file1)

        # Process the DataFrame
        df = process_dataframe(df)

        # Ensure the required feature columns are in the DataFrame
        if not all(col in df.columns for col in FEATURE_COLUMNS):
            return 'Missing feature columns in the uploaded file', 400

        # Select feature columns for prediction
        df_features = df[FEATURE_COLUMNS]
        df_preprocessed = preprocess_data(df_features)

        # Make predictions
        predictions = model.predict(df_preprocessed)
        df['Predicted'] = predictions

        # Save to an Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)

        return send_file(output, download_name='predictions.xlsx', as_attachment=True)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)

