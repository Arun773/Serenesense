from flask import Flask, request, jsonify,render_template, send_from_directory
import speech_recognition as sr
import speech
app = Flask(__name__)

def detect_stress(message):  
    if 'overwhelmed' in message or 'anxious' in message or 'stressed' in message or 'panic' in message or 'depressed' in message or 'because' in message or 'state' in message or 'nervous' in message or 'unease' in message or 'depression' in message:
        return "high"
    elif 'hopeless' in message or 'alone' in message or 'stuck' in message or 'sad' in message or 'helpless' in message or 'tired' in message or 'empty' in message:
        return "high."
    elif 'helplessness' in message or 'burden' in message or 'feeling' in message or 'angry' in message or 'blent' in message or 'feelingwell' in message or 'suffering' in message or 'emptiness' in message or 'void' in message:
        return "high.."
    elif 'busy' in message or 'pressure' in message or 'worried' in message or 'tension' in message or 'nervous' in message or 'discomfort' in message or 'strain' in message or 'uneasy' in message:
        return "moderate"
    else:
        return "low"
    
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/speech-to-text')
def speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Listen for user input
        audio_data = recognizer.listen(source)

        print("Processing...")

        try:
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
            return text  # Return the recognized text
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
            return "Sorry, could not understand audio."
        except sr.RequestError as e:
            print("Error fetching results; {0}".format(e))
            return "Error fetching results: {0}".format(e)

@app.route('/audio')
def audio():
    return render_template('audioTherapy.html')

@app.route('/detect_stress_audio', methods=['POST'])
def detect_stress_level_audio():
    data = request.form['stressContent']
    message = data.lower()
    stress_level = detect_stress(message)
    if stress_level == "high":
        reply = "Your stress levels seem elevated. Watch this standup comedy for instant relief: <a href='https://www.youtube.com/watch?v=Tqsz6fjvhZM' target='_blank'><b>Standup Comedy Video</b></a> or check out some <a href='https://www.healthline.com/nutrition/laughing-yoga' target='_blank'><b>Laughter Yoga</b></a>."
    elif stress_level == "high.":
        reply = "Your stress levels seem elevated. Enjoy these memes to lighten your mood: <a href='/static/images/laughing/meme1.jpg' target='_blank'><b>Meme 1</b></a> | <a href='/static/images/laughing/meme2.jpg' target='_blank'><b>Meme 2</b></a>. Or try <a href='https://www.healthline.com/nutrition/laughing-yoga' target='_blank'><b>Laughter Yoga</b></a>."
    elif stress_level == "high..":
        reply = "Your stress levels seem elevated. Try watching this funny video: <a href='https://www.youtube.com/watch?v=Y2Oj9gllHno' target='_blank'><b>Standup Video</b></a> or browse our memes section for a quick laugh!"
    elif stress_level == "moderate":
        reply = "You have a moderate level of stress. Take a break and enjoy some laughter yoga: <a href='https://www.healthline.com/nutrition/laughing-yoga' target='_blank'><b>Laughter Yoga Guide</b></a> or watch a short standup: <a href='https://www.youtube.com/watch?v=XDlyS4N__3o' target='_blank'><b>Standup Clip</b></a>."
    elif stress_level == "low":
        reply = "Your stress level seems low. Keep smiling! Enjoy a meme: <a href='/static/images/laughing/meme3.jpg' target='_blank'><b>Meme 3</b></a> or share a joke with a friend!"
    else:
        reply = "Unable to detect stress level from the provided message."
    return jsonify({'stress_level': stress_level, 'recommendation': reply})

@app.route('/reading')
def reading():
    return render_template('readingTherapy.html')

@app.route('/detect_stress_read', methods=['POST'])
def detect_stress_level_read():
    data = request.form['stressContent']
    message = data.lower()
    stress_level = detect_stress(message)
    if stress_level == "high":
        reply = "Your stress levels seem elevated. We suggest accessing a Book to manage stress and induce relaxation.:<iframe src='https://ahigherthought.com/the-best-way-to-refresh-mind-body-soul/' frameborder='0' width='100%' height='200px'></iframe>"
    elif stress_level == "high.":
        reply = "Your stress levels seem elevated. We suggest accessing a Book to manage stress and induce relaxation.:<iframe src='https://www.developgoodhabits.com/inspirational-stories/' frameborder='0' width='100%' height='200px'></iframe>"
    elif stress_level == "high..":
        reply = "Your stress levels seem elevated. We suggest accessing a Book to manage stress and induce relaxation.:<iframe src='https://www.samuelthomasdavies.com/book-summaries/self-help/atomic-habits/' frameborder='0' width='100%' height='200px'></iframe>"    
    elif stress_level == "moderate":
        reply = "Your current condition suggests a moderate level of stress. We advise considering a Book as a potential intervention to mitigate its effects.:<iframe src='https://jamesclear.com/book-summaries/the-subtle-art-of-not-giving-a-fck' frameborder='0' width='100%' height='200px'></iframe>"
    elif stress_level == "low":
        reply = "Your stress level seems low. Enjoy this uplifting Book: <iframe src='https://www.sloww.co/ikigai-book/' frameborder='0' width='100%' height='200px'></iframe>"
    else:
        reply = "Unable to detect stress level from the provided message."
    return jsonify({'stress_level': stress_level, 'recommendation': reply})

@app.route('/yoga')
def yoga():
    return render_template('yogatherapy.html')

@app.route('/detect_stress_yoga', methods=['POST'])
def detect_stress_level_yoga():
    data = request.form['stressContent']
    message = data.lower()
    stress_level = detect_stress(message)
    if stress_level == "high":
        reply = "Your stress levels seem elevated.\n We suggest 'ARDHA CHAKRASANA' to manage stress and induce relaxation:<iframe width='560' height='315' src='https://www.youtube.com/embed/97kzExeOMRs?si=xikN0LVjpXfjRGE0' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "high.":
        reply = "Your stress levels seem elevated.\n We suggest 'VIRABADRASANA' to manage stress and induce relaxation:<iframe width='560' height='315' src='https://www.youtube.com/embed/fniJmxMyqBU?si=XK1OAI_kfwRvPDm6&amp;start=26' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "high..":
        reply = "Your stress levels seem elevated.\n We suggest 'KONASANA' to manage stress and induce relaxation:<iframe width='560' height='315' src='https://www.youtube.com/embed/PwHyARpmbRI?si=ozvuHnIUk0iCiLpo' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "moderate":
        reply = "Your current condition suggests a moderate level of stress.\n We advise considering 'TRIKONASANA' as a potential intervention to mitigate its effects.:<iframe width='560' height='315' src='https://www.youtube.com/embed/S6gB0QHbWFE?si=itM0g5h4LxW1i7JA' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "low":
        reply = "Your stress level seems low. We advise 'PADMASANA': <iframe width='560' height='315' src='https://www.youtube.com/embed/UTOBheDjLhQ?si=fx0dM53mNmtWnwct' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    else:
        reply = "Unable to detect stress level from the provided message."
    return jsonify({'stress_level': stress_level, 'recommendation': reply})

@app.route('/laugh')
def laugh():
    return render_template('laughTherapy.html')

@app.route('/detect_stress_laugh', methods=['POST'])
def detect_stress_level_laugh():
    data = request.form['stressContent']
    message = data.lower()
    stress_level = detect_stress(message)
    if stress_level == "high":
        reply = "Your stress levels seem elevated.\n We suggest Standups to manage stress:<iframe width='560' height='315' src='https://www.youtube.com/embed/2Oy4HpUJSgE?si=XWrsQkSfCWJn2YpG' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "high.":
        reply = "Your stress levels seem elevated.\n We suggest Standups to manage stress:<iframe width='560' height='315' src='https://www.youtube.com/embed/DLgtsl6RLWE?si=a6LFZMxgOe8UzZxq' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "high..":
        reply = "Your stress levels seem elevated.\n We suggest Standups to manage stress:<iframe width='560' height='315' src='https://www.youtube.com/embed/QhMO5SSmiaA?si=G9KAwo-kBYVWjIL7' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "moderate":
        reply = "Your current condition suggests a moderate level of stress.\n We advise considering Standups as a potential intervention to mitigate its effects:<iframe width='560' height='315' src='https://www.youtube.com/embed/0xmNtymiGK8?si=zJsHbly5JaCNCqSw' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    elif stress_level == "low":
        reply = "Your stress level seems low. We advise Standups: <iframe width='560' height='315' src='https://www.youtube.com/embed/LJ3n1RHqFlE?si=MZhbQJbu0yhgyA23' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share' allowfullscreen></iframe>"
    else:
        reply = "Unable to detect stress level from the provided message."
    return jsonify({'stress_level': stress_level, 'recommendation': reply})

@app.route('/talking')
def talking():
    return render_template('talkingTherapy.html')

# @app.route('/detect_stress_talking', methods=['POST'])
# def detect_stress_level_talk():
#     data = request.form['stressContent']
#     message = data.lower()
#     stress_level = detect_stress(message)
#     if stress_level == "high":
#         reply = "It seems like you're experiencing high levels of stress. Here's a Spotify playlist that may help: [Playlist Name] - [https://open.spotify.com/embed/playlist/37i9dQZF1DWXe9gFZP0gtP?utm_source=generator]"
#     elif stress_level == "moderate":
#         reply = "You might be experiencing moderate stress. Check out this calming track on Spotify: [Track Name] - [Artist Name] - [Track Link]"
#     elif stress_level == "low":
#         reply = "Your stress level seems low. Enjoy this uplifting song on Spotify: [Track Name] - [Artist Name] - [Track Link]"
#     else:
#         reply = "Unable to detect stress level from the provided message."
#     return jsonify({'stress_level': stress_level, 'recommendation': reply})

@app.route('/child')
def child():
    return render_template('childTherapy.html')


@app.route('/spirtual')
def spirtual():
    return render_template('spirtualTherapy.html')


@app.route('/special')
def special():
    return render_template('specialTherapy.html')

@app.route('/login')
def login():
    return render_template('login.html')


if __name__ == '__main__':
    app.run(debug=True)