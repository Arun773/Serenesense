import numpy as np
from deepface import DeepFace
from PIL import Image
import tempfile

def detect_face_emotion(image_file):
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
        img = Image.open(image_file)
        img.save(temp.name)
        temp_path = temp.name
    try:
        analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        emotion = analysis['dominant_emotion']
        confidence = analysis['emotion'][emotion]
    except Exception as e:
        return {"error": str(e)}
    return {"emotion": emotion, "confidence": float(confidence) / 100.0} 