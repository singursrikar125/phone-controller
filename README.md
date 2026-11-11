# 📱 Phone Controller

A web-based Android app launcher powered by Python, Scikit-learn (TF-IDF + Logistic Regression), Flask, and ADB.

## Features
- **Action Page:** Type natural language commands (e.g., *"open settings"*, *"play song on youtube"*) to launch apps on a connected Android device.
- **Code Page:** View the underlying Machine Learning intent classification code directly in the browser.

## Setup & Usage
1. Enable **USB Debugging** in Developer Options on your Android phone and connect it to your PC.
2. Verify connection by running `adb devices` in your terminal.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   python app.py
   ```
5. Open `http://127.0.0.1:5000` in your browser.
