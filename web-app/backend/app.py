from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from predict_audio import predict_from_audio_clip

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"Error": "No audio file uploaded!"}), 400
    try:
        audio_file = request.files['audio']
        result = predict_from_audio_clip(audio_file)
        return jsonify(result)
    except Exception as e:
        error_msg = str(e)
        return jsonify({"Error": error_msg}), 500

if __name__ == '__main__':
    app.run(port=5050)
