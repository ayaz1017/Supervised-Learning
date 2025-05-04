from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Function to preprocess input
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Encode categorical variables
    for column, encoder in encoders.items():
        if column in df:
            try:
                df[column] = encoder.transform(df[column].astype(str))
            except ValueError:
                df[column] = encoder.transform([encoder.classes_[0]])[0]  # Default category

    # Ensure input matches model features
    df = df.reindex(columns=feature_names, fill_value=0)
    df = df.astype(float)

    return df

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {key: request.form[key] for key in request.form}

        # Convert numerical fields
        input_data['SeniorCitizen'] = int(input_data['SeniorCitizen'])
        input_data['tenure'] = int(input_data['tenure'])
        input_data['MonthlyCharges'] = float(input_data['MonthlyCharges'])

        # Preprocess input
        input_df = preprocess_input(input_data)

        # Make prediction
        prediction = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df).tolist()

        return jsonify({
            "Prediction": "Churn" if prediction == 1 else "No Churn",
            "Probability": pred_prob
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)





