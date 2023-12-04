from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
 

from flask import Flask, request, jsonify
import numpy as np
import librosa  # You may need to install this library: pip install librosa
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load("/workspaces/Model-deploy/models/rf_model.pkl")

def extract_audio_features(audio_file_path):
   mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
   mfccs_processed = np.mean(mfccs.T, axis=0)
   # Normalize the features
   scaled_features = scaler.transform([mfccs_processed])
   return  scaled_features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get audio file path from the request
        audio_file_path = request.json['audio_file_path']

        # Extract audio features
        audio_features = extract_audio_features(audio_file_path)

        # Make predictions using the loaded model
        predictions = model.predict([audio_features])

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
