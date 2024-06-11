from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load data for preprocessing and label encoding
data = pd.read_csv('StudentsPerformance.csv')

# Preprocess data to get the same label encoder used during training
y = data['parental level of education']  # Target variable
label_encoder = LabelEncoder()
label_encoder.fit(y)

# Load trained model
model = joblib.load('finalized_model.sav')

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        features = [float(x) for x in request.form.values()]
        
        # Make prediction using the loaded model
        prediction = model.predict([features])[0]
        
        # Decode the predicted label if needed
        prediction = label_encoder.inverse_transform([prediction])[0]
        
        # Return the prediction result
        return f"The predicted parental level of education is: {prediction}"

if __name__ == '__main__':
    app.run(debug=True)