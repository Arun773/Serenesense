from flask import Flask, request, jsonify, render_template
from models.face_emotion import detect_face_emotion
from models.voice_emotion import detect_voice_emotion
from models.text_sentiment import detect_text_sentiment
from models.fusion import fuse_predictions
from models.recommend import recommend_therapy
from utils.db import store_result, fetch_results
import os


app = Flask(__name__, template_folder='Templates', static_folder='static')

# ---------- Frontend routes ----------
@app.get('/')
def home_page():
    return render_template('index.html')

@app.get('/login')
def login_page():
    return render_template('login.html')

@app.get('/audio')
def audio_page():
    return render_template('audioTherapy.html')

@app.get('/reading')
def reading_page():
    return render_template('readingTherapy.html')

@app.get('/yoga')
def yoga_page():
    return render_template('yogatherapy.html')

@app.get('/laugh')
def laugh_page():
    return render_template('laughTherapy.html')

@app.get('/talking')
def talking_page():
    return render_template('talkingTherapy.html')

@app.get('/child')
def child_page():
    return render_template('childTherapy.html')

# Note: template uses 'spirtual' spelling, preserve route for compatibility
@app.get('/spirtual')
def spirtual_page():
    return render_template('spirtualTherapy.html')

@app.get('/special')
def special_page():
    return render_template('specialTherapy.html')

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
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    print(f"Starting Serenesense on port {port}")
    app.run(debug=True, port=port)