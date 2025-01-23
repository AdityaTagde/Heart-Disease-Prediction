from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model (ensure this path is correct)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route that accepts form data and returns prediction result
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting features from the submitted form
        features = [
            int(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            int(request.form['trestbps']),
            int(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            int(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]
        
        # Convert features into a numpy array (shape must be (1, 13) for prediction)
        features = np.array(features).reshape(1, -1)

        # Get the prediction from the model
        prediction = model.predict(features)

        # Prediction result (0 or 1)
        result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'

        # Suggested actions based on prediction
        suggestions = []
        if prediction[0] == 1:
            suggestions = [
                "Consult a doctor immediately for further evaluation.",
                "Follow a heart-healthy diet.",
                "Increase physical activity and exercise.",
                "Monitor cholesterol and blood pressure regularly."
            ]
        else:
            suggestions = [
                "Maintain a healthy lifestyle with regular check-ups.",
                "Focus on a balanced diet and regular exercise.",
                "Continue monitoring heart health."
            ]

        # Render result page with prediction and suggestions
        return render_template('result.html', prediction=result, suggestions=suggestions)

    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
