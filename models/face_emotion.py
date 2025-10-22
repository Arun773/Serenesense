import numpy as np
from deepface import DeepFace
from PIL import Image
import tempfile
import argparse
import json
import sys

def detect_face_emotion(image_file):
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
        img = Image.open(image_file)
        # Ensure image is RGB (remove alpha) for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(temp.name)
        temp_path = temp.name
    try:
        analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        # DeepFace may return a list (per-face) or a dict; normalize to dict
        if isinstance(analysis, list) and len(analysis) > 0:
            analysis = analysis[0]
        emotion = analysis.get('dominant_emotion')
        emotions_dict = analysis.get('emotion', {})
        confidence = emotions_dict.get(emotion)
        if confidence is None:
            raise ValueError("Could not extract confidence for detected emotion")
    except Exception as e:
        return {"error": str(e)}
    # Normalize confidence to 0..1
    conf_val = float(confidence)
    if conf_val > 1.0:
        conf_val = conf_val / 100.0
    return {"emotion": emotion, "confidence": conf_val} 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run face emotion detection on an image using DeepFace.")
    parser.add_argument("image_path", help="Path to the image file (jpg/png/jpeg)")
    args = parser.parse_args()

    try:
        with open(args.image_path, "rb") as f:
            result = detect_face_emotion(f)
        # Print JSON to stdout
        print(json.dumps(result, indent=2))
        # Non-zero exit if error
        if "error" in result:
            sys.exit(2)
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {args.image_path}"}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)