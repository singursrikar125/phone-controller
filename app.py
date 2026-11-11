from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import subprocess

app = Flask(__name__)

# --- 1. TRAINING DATA ---
sentences = [
    "open youtube", "play song on youtube", "start youtube",
    "open whatsapp", "start whatsapp", "launch whatsapp",
    "open settings", "start settings", "phone settings",
    "open chatgpt", "start chatgpt", "launch chatgpt",
    "open chrome", "start chrome", "launch browser",
    "open playstore", "open play store", "start playstore",
    "open gmail", "check gmail", "start gmail",
    "open maps", "open google maps", "start maps",
    "open drive", "open google drive", "start drive",
    "open instagram", "start instagram", "launch instagram",
    "open music", "play music", "start music",
    "open camera", "start camera", "launch camera",
    "open video", "open gallery", "play video",
    "open files", "open file manager", "show files",
    "open phone", "open dialer", "start phone",
    "open messages", "open sms", "start messages",
    "open notes", "open keep notes", "start notes"
]

intents = [
    "youtube",   "youtube",   "youtube",
    "whatsapp",  "whatsapp",  "whatsapp",
    "settings",  "settings",  "settings",
    "chatgpt",   "chatgpt",   "chatgpt",
    "chrome",    "chrome",    "chrome",
    "playstore", "playstore", "playstore",
    "gmail",     "gmail",     "gmail",
    "maps",      "maps",      "maps",
    "drive",     "drive",     "drive",
    "instagram", "instagram", "instagram",
    "music",     "music",     "music",
    "camera",    "camera",    "camera",
    "video",     "video",     "video",
    "files",     "files",     "files",
    "phone",     "phone",     "phone",
    "messages",  "messages",  "messages",
    "notes",     "notes",     "notes"
]

app_packages = {
    "youtube": "com.google.android.youtube",
    "whatsapp": "com.whatsapp",
    "settings": "com.android.settings",
    "chatgpt": "com.openai.chatgpt",
    "chrome": "com.android.chrome",
    "playstore": "com.android.vending",
    "gmail": "com.google.android.gm",
    "maps": "com.google.android.apps.maps",
    "drive": "com.google.android.apps.docs",
    "instagram": "com.instagram.android",
    "music": "com.google.android.apps.youtube.music",
    "camera": "com.android.camera",
    "video": "com.google.android.apps.photos",
    "files": "com.google.android.documentsui",
    "phone": "com.android.dialer",
    "messages": "com.google.android.apps.messaging",
    "notes": "com.google.android.keep"
}

# --- 2. TRAIN ML MODEL ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X, intents)

# --- 3. WEBSITE ROUTES ---
@app.route("/")
def action_page():
    return render_template("action.html", apps=list(app_packages.keys()))

@app.route("/code")
def code_page():
    # Reads this file so the website displays the live Python code
    with open(__file__, "r", encoding="utf-8") as f:
        source_code = f.read()
    return render_template("code.html", code=source_code)

@app.route("/run-command", methods=["POST"])
def run_command():
    data = request.get_json()
    user_command = data.get("command", "").lower().strip()

    if not user_command:
        return jsonify({"status": "error", "message": "Please enter a command."})

    X_test = vectorizer.transform([user_command])
    predicted_intent = model.predict(X_test)[0]
    confidence = float(model.predict_proba(X_test)[0].max())

    # Threshold lowered slightly to 0.25 to account for 17 classes
    if confidence > 0.25 and predicted_intent in app_packages:
        package_name = app_packages[predicted_intent]
        try:
            subprocess.run(
                ["adb", "shell", "monkey", "-p", package_name, "1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return jsonify({
                "status": "success",
                "message": f"Opening {predicted_intent.title()}...",
                "confidence": round(confidence * 100, 1)
            })
        except Exception as e:
            return jsonify({"status": "error", "message": f"ADB Error: {str(e)}"})
    else:
        return jsonify({
            "status": "error",
            "message": "Sorry, I couldn't understand which app to open."
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
