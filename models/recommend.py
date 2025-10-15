def recommend_therapy(data):
    stress = data.get("stress_level", "low")
    emotion = data.get("emotion", "neutral")
    if stress == "high":
        therapy = "laughter_yoga"
        resources = [
            "https://www.healthline.com/nutrition/laughing-yoga",
            "https://www.youtube.com/watch?v=XDlyS4N__3o"
        ]
    elif stress == "moderate":
        therapy = "reading_therapy"
        resources = [
            "https://www.sloww.co/ikigai-book/"
        ]
    else:
        therapy = "yoga"
        resources = [
            "https://www.youtube.com/watch?v=UTOBheDjLhQ"
        ]
    return {"therapy": therapy, "resources": resources} 