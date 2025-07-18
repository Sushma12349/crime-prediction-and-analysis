from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'static', 'crimes.csv')
df = pd.read_csv(csv_path)

# Ensure 'hour' exists
if 'hour' not in df.columns:
    import numpy as np
    df['hour'] = np.random.randint(0, 24, size=len(df))

# Encode crime type
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])

# Train model
X = df[['latitude', 'longitude', 'hour']]
y = df['type_encoded']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/')
def index():
    crimes_dict = df.to_dict(orient='records')
    return render_template('map.html', crimes=crimes_dict)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    hour = data.get('hour')
    prediction = model.predict([[lat, lon, hour]])[0]
    predicted_label = le.inverse_transform([prediction])[0]
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    import webbrowser
    from threading import Timer
    port = 5000
    url = f"http://127.0.0.1:{port}"
    Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(debug=True, port=port)
