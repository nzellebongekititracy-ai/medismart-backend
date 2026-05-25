from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
import joblib
import os 

app = Flask(__name__)
# Enable CORS so your front-end or mobile application can make requests to this API
CORS(app) 

# Paths to your model brain and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_distress_model.onnx')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data_scaler.pkl')

# Load the lightweight ONNX runtime inference session and the scaler
session = ort.InferenceSession(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Get the internal input node name required by the ONNX model structure
input_name = session.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract vital signs sent from the request body
        data = request.json
        hr = data['hr']
        spo2 = data['spo2']
        temp = data['temp']

        # 1. Structure the raw data into a 2D numpy array for scaling
        input_data = np.array([[hr, spo2, temp]], dtype=np.float32)
        scaled_input = scaler.transform(input_data)
        
        # 2. Reshape the array to match the LSTM time-series input format:
        # (batch_size = 1, timesteps = 1, features = 3)
        final_input = np.reshape(scaled_input, (1, 1, 3)).astype(np.float32)

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
    # Bind to the PORT environment variable assigned dynamically by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
