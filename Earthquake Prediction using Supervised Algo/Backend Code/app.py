from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('earthquake_model_sm.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        latitude = float(data.get('Latitude(deg)', 0))
        longitude = float(data.get('Longitude(deg)', 0))
        depth = float(data.get('Depth(km)', 0))
        stations = float(data.get('No_of_Stations', 0))

        features = np.array([[latitude, longitude, depth, stations]])
        prediction = model.predict(features)
        predicted_magnitude = float(prediction[0])

        earthquake_chance = "Yes" if predicted_magnitude >= 4.0 else "No"

        return jsonify({
            'prediction': predicted_magnitude,
            'earthquake_chance': earthquake_chance
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)



