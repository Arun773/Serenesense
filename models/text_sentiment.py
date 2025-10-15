from transformers import pipeline

# Load sentiment analysis pipeline (DistilBERT)
sentiment_pipeline = pipeline("sentiment-analysis")

def detect_text_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()
    score = result['score']
    # Map sentiment to stress level
    if label == 'negative':
        stress_level = 'high'
    elif label == 'neutral':
        stress_level = 'moderate'
    else:
        stress_level = 'low'
    return {"sentiment": label, "stress_level": stress_level, "confidence": score} 