import flask
import pickle
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
try:
    with open("/workspaces/Model-deploy/models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(e)


def extract_audio_features(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Compute the mean of the MFCCs
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.transform([mfccs_processed])

    return scaled_features


app = flask.Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # Get the audio file from the request
    audio_file = flask.request.files["audio_file"]

    # Extract audio features
    audio_features = extract_audio_features(audio_file)

    # Make a prediction using the model
    prediction = model.predict(audio_features)

    # Return the prediction
    return flask.jsonify({"prediction": prediction[0]})


@app.route("/get_prediction_output", methods=["POST"])
def get_prediction_output():
    # Get the data from the request
    data = flask.request.get_json()

    # Make a prediction using the data
    prediction = predict_mpg(data)

    # Return the prediction
    return flask.jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)