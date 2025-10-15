def fuse_predictions(face=None, voice=None, text=None):
    # Assign weights (can be tuned)
    weights = {"face": 0.35, "voice": 0.30, "text": 0.35}
    levels = {"high": 2, "moderate": 1, "low": 0}
    votes = []
    confidences = []
    if face:
        votes.append(levels.get(face.get("emotion", "low"), 0) * weights["face"])
        confidences.append(face.get("confidence", 0))
    if voice:
        votes.append(levels.get(voice.get("emotion", "low"), 0) * weights["voice"])
        confidences.append(voice.get("confidence", 0))
    if text:
        votes.append(levels.get(text.get("stress_level", "low"), 0) * weights["text"])
        confidences.append(text.get("confidence", 0))
    avg_vote = sum(votes) / sum(weights.values()) if votes else 0
    if avg_vote >= 1.5:
        final_level = "high"
    elif avg_vote >= 0.5:
        final_level = "moderate"
    else:
        final_level = "low"
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    return {"final_stress_level": final_level, "confidence": round(avg_conf, 2), "details": {"face": face, "voice": voice, "text": text}} 