from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files['image']
    img_np = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({'roundness': 0, 'message': 'No chapatti found 😢'})

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)

    roundness = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0
    roundness_percent = round(roundness * 100, 2)

    if roundness_percent >= 85:
        message = "Perfectly round chapatti! 🟢 Serve it hot!"
    elif roundness_percent >= 65:
        message = "Almost there! Try rolling evenly. 🟡"
    else:
        message = "Chapatti needs love. Practice more! 🔴"

    return jsonify({
        'roundness': roundness_percent,
        'message': message
    })

if __name__ == '__main__':
    app.run(debug=True)
