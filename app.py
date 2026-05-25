from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import os 

app = Flask(__name__)
CORS(app) 

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_distress_model.h5')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data_scaler.pkl')

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        hr = data['hr']
        spo2 = data['spo2']
        temp = data['temp']

        input_data = np.array([[hr, spo2, temp]])
        scaled_input = scaler.transform(input_data)
        
        final_input = np.reshape(scaled_input, (1, 1, 3))

        prediction = model.predict(final_input)
        probability = float(prediction[0][0])
        
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
