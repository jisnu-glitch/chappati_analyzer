from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

port = int(os.environ.get('PORT', 10000))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files['image']
    img_np = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Step 1: Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Step 2: Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Morphological operations to clean noise
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({'roundness': 0, 'message': 'No chapatti found 😢'})

    # Step 5: Filter contours by area and circularity
    best_contour = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue  # Skip small shapes
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        roundness = 4 * np.pi * (area / (perimeter * perimeter))
        if roundness > best_score:
            best_score = roundness
            best_contour = cnt

    if best_contour is None:
        return jsonify({'roundness': 0, 'message': 'No valid chapatti shape found 😢'})

    # Final roundness score
    roundness_percent = round(best_score * 100, 2)

    # Grading
    if roundness_percent >= 95:
        message = "That’s a *textbook* chapatti! 🟢 Chef’s kiss 👨‍🍳"
    elif roundness_percent >= 85:
        message = "Perfectly round chapatti! 🟢 Serve it hot!"
    elif roundness_percent >= 65:
        message = "Almost there! Try rolling evenly. 🟡"
    elif roundness_percent >= 40:
        message = "Chapatti shape needs practice! 🟠"
    else:
        message = "Chapatti doesn’t resemble a circle 🟥 Try again!"

    return jsonify({
        'roundness': roundness_percent,
        'message': message
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

app.run(host='0.0.0.0', port=port)
