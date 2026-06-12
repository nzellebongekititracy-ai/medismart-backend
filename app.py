from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import os 

app = Flask(__name__)
CORS(app) 

# Paths to your model brain and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_distress_model.onnx')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data_scaler.pkl')

# SAFE MULTI-THREADING ENGINE CONFIGURATION FOR GUNICORN (Fixes Code 139)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Load inference session with safe execution configurations
session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
scaler = joblib.load(SCALER_PATH)
input_name = session.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract vital signs sent from the request body
        data = request.json
        hr = float(data['hr'])
        spo2 = float(data['spo2'])
        temp = float(data['temp'])

        # 1. Structure the raw data into a standard 2D array matrix for scaling
        input_data = np.array([[hr, spo2, temp]], dtype=np.float32)
        
        # Use columns matching your training structure to satisfy MinMaxScaler requirements
        if hasattr(scaler, "feature_names_in_"):
            input_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
        else:
            input_df = pd.DataFrame(input_data, columns=['hr', 'spo2', 'temp'])
            
        scaled_input = scaler.transform(input_df)
        
        # 2. Match the exact 10 timesteps required by your LSTM network
        sequence_input = np.repeat(scaled_input, 10, axis=0) 
        final_input = np.reshape(sequence_input, (1, 10, 3)).astype(np.float32)

        # 3. Run the prediction through the ONNX execution engine
        prediction = session.run(None, {input_name: final_input})
        probability = float(prediction[0][0][0])
        
        # 4. Determine classification threshold
        status = "Distress" if probability > 0.5 else "Stable"

        return jsonify({
            "status": status,
            "probability": round(probability, 4),
            "message": "Analysis completed successfully"
        })

    except KeyError as ke:
        return jsonify({"error": f"Missing required vital parameter: {str(ke)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
