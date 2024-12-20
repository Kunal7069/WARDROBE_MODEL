from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import random

app = Flask(__name__)
CORS(app)  
port=5000

# Define input feature names and output target names
columns_X = ['Outfit', 'Style', 'Belt', 'Watch']
columns_y = ['Shoes', 'Shirt', 'Pants']

# Load pre-trained models and encoders
loaded_models = {col: joblib.load(f"{col}_logistic_model.pkl") for col in columns_y}
loaded_label_encoders_X = {col: joblib.load(f"{col}_encoder.pkl") for col in columns_X}
loaded_label_encoders_y = {col: joblib.load(f"{col}_output_encoder.pkl") for col in columns_y}

# Function to validate and correct manual input
def validate_and_correct_input(manual_input):
    corrected_input = {}
    for col, value in manual_input.items():
        # Check if the value matches one of the recognized classes
        if value.upper() in [cls.upper() for cls in loaded_label_encoders_X[col].classes_]:
            corrected_input[col] = next(cls for cls in loaded_label_encoders_X[col].classes_ if cls.upper() == value.upper())
        else:
            raise ValueError(f"Invalid value '{value}' for column '{col}'. Expected one of: {loaded_label_encoders_X[col].classes_}")
    return corrected_input

# API Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        manual_input = request.json
        belt_input=manual_input.get('Belt')
        if belt_input=='YES':
            random_variable = random.randint(1, 10)
            if random_variable%2==0:
                manual_input['Belt']='BLACK'
            else:
                manual_input['Belt']='BROWN'
        corrected_input = validate_and_correct_input(manual_input)
        
        # Encode the input
        encoded_input = {col: loaded_label_encoders_X[col].transform([corrected_input[col]])[0] for col in corrected_input}
        
        # Predict outputs
        predictions = {}
        for col, model in loaded_models.items():
            prediction = model.predict(pd.DataFrame([encoded_input]))
            predictions[col] = loaded_label_encoders_y[col].inverse_transform(prediction)[0]
        
        if belt_input=='NO':
            predictions['Belt']='NO'
        elif predictions['Pants']=='BROWN':
            predictions['Belt']='BROWN'
        else:
            predictions['Belt']='BLACK'
        
        predictions['Watch']=corrected_input['Watch']
            
        # Return the predictions as JSON
        return jsonify({
            "status": "success",
            "predictions": predictions
        })
    
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)


