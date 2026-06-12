from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import joblib
import os 

app = Flask(__name__)
CORS(app) 

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_distress_model.onnx')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data_scaler.pkl')

session = ort.InferenceSession(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
input_name = session.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    # NO TRY/CATCH BLOCK HERE — FORCE THE ERROR TO SHOW IN THE LOGS
    data = request.json
    print("--- INCOMING DATA FROM MOBILE ---", data)
    
    hr = float(data['hr'])
    spo2 = float(data['spo2'])
    temp = float(data['temp'])

    input_data = np.array([[hr, spo2, temp]], dtype=np.float32)
    
    # We use a clean DataFrame to match training shapes exactly
    import pandas as pd
    if hasattr(scaler, "feature_names_in_"):
        input_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
    else:
        input_df = pd.DataFrame(input_data, columns=['hr', 'spo2', 'temp'])
        
    scaled_input = scaler.transform(input_df)
    
    # Reshape to match LSTM layout
    final_input = np.reshape(scaled_input, (1, 1, 3)).astype(np.float32)
    print("--- SHAPE SENT TO ONNX ---", final_input.shape)

    prediction = session.run(None, {input_name: final_input})
    probability = float(prediction[0][0][0])
    
    status = "Distress" if probability > 0.5 else "Stable"

    return jsonify({
        "status": status,
        "probability": round(probability, 4)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
