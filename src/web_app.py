from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

# Create Flask app with correct paths
app = Flask(__name__,
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)

# Initialize CORS
CORS(app)

# Emotion labels matching your 8-class model
EMOTION_LABELS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# Load model
try:
    model_path = os.path.join(PROJECT_ROOT, 'model', 'Best_model.keras')
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Import utilities from src.utils
from src.utils import preprocess_face, detect_faces

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests."""
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.json['image']
        # Extract base64 image data
        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect faces
        faces = detect_faces(frame)

        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess and predict
            processed_face = preprocess_face(face_roi)
            predictions = model.predict(processed_face, verbose=0)

            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            emotion_label = EMOTION_LABELS[emotion_idx]

            results.append({
                'emotion': emotion_label,
                'confidence': confidence,
                'bbox': [int(x), int(y), int(w), int(h)],
                'all_predictions': {EMOTION_LABELS[i]: float(pred) for i, pred in enumerate(predictions[0])}
            })

        return jsonify({'faces': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # You can specify host='0.0.0.0' to make the app accessible from other machines
    app.run(debug=True, host='127.0.0.1')