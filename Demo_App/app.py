from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load data
data = pd.read_csv('StudentsPerformance.csv')

# Preprocess data
X = data[['math score', 'reading score', 'writing score']]  # Features
y = data['parental level of education']  # Target variable

# Encoding categorical variables if any
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Load model
model = joblib.load('finalized_model.sav')
model.fit(X, y)
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