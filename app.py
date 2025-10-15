from flask import Flask, request, jsonify
from models.face_emotion import detect_face_emotion
from models.voice_emotion import detect_voice_emotion
from models.text_sentiment import detect_text_sentiment
from models.fusion import fuse_predictions
from models.recommend import recommend_therapy
from utils.db import store_result, fetch_results
import os


app = Flask(__name__)

@app.route('/api/face-emotion', methods=['POST'])
def api_face_emotion():
    image = request.files['image']
    result = detect_face_emotion(image)
    return jsonify(result)

@app.route('/api/voice-emotion', methods=['POST'])
def api_voice_emotion():
    audio = request.files['audio']
    result = detect_voice_emotion(audio)
    return jsonify(result)

@app.route('/api/text-sentiment', methods=['POST'])
def api_text_sentiment():
    text = request.json['text']
    result = detect_text_sentiment(text)
    return jsonify(result)

@app.route('/api/stress-fusion', methods=['POST'])
def api_stress_fusion():
    image = request.files.get('image')
    audio = request.files.get('audio')
    text = request.form.get('text')
    face_result = detect_face_emotion(image) if image else None
    voice_result = detect_voice_emotion(audio) if audio else None
    text_result = detect_text_sentiment(text) if text else None
    result = fuse_predictions(face=face_result, voice=voice_result, text=text_result)
    return jsonify(result)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.json
    result = recommend_therapy(data)
    return jsonify(result)

@app.route('/api/results', methods=['GET'])
def api_results():
    user_id = request.args.get('user_id')
    results = fetch_results(user_id)
    return jsonify(results.data)

if __name__ == '__main__':
    app.run(debug=True)