from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import joblib
import os 

app = Flask(__name__)
CORS(app) 

# Paths to your model brain and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_distress_model.onnx')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data_scaler.pkl')

# Load the ONNX runtime session and the scaler
session = ort.InferenceSession(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Get the input name required by the ONNX model structure
input_name = session.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        hr = data['hr']
        spo2 = data['spo2']
        temp = data['temp']

        # Process the raw vital signs exactly like training
        input_data = np.array([[hr, spo2, temp]], dtype=np.float32)
        scaled_input = scaler.transform(input_data)
        
        # Shape the array for the LSTM sequence structure (1 sample, 1 timestep, 3 features)
        final_input = np.reshape(scaled_input, (1, 1, 3)).astype(np.float32)

        # Run inference using the ONNX session
        prediction = session.run(None, {input_name: final_input})
        probability = float(prediction[0][0][0])
        
        status = "Distress" if probability > 0.5 else "Stable"

        return jsonify({
            "status": status,
            "probability": round(probability, 4),
            "message": "Analysis successful"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
