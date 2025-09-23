import cv2
import numpy as np

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

# Colors for each emotion
EMOTION_COLORS = {
    'Anger': (0, 0, 255),        # Red
    'Contempt': (128, 0, 0),     # Maroon
    'Disgust': (0, 128, 0),      # Green
    'Fear': (128, 0, 128),       # Purple
    'Happy': (0, 255, 255),      # Yellow
    'Neutral': (128, 128, 128),  # Gray
    'Sad': (255, 0, 0),          # Blue
    'Surprise': (0, 165, 255),   # Orange
}

def preprocess_face(face_roi, target_size=(96, 96)):
    """Preprocess face ROI for emotion prediction"""
    # Resize to model input size
    face_resized = cv2.resize(face_roi, target_size)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    processed_face = face_rgb / 255.0
    
    # Add a batch dimension
    return np.expand_dims(processed_face, axis=0)

def detect_faces(frame):
    """Detect faces in a frame using a Haar cascade classifier."""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"Error: Face cascade classifier not loaded at {cascade_path}")
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def draw_emotion_info(frame, x, y, w, h, emotion, confidence, all_predictions=None):
    """Draw emotion information on the frame"""
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    
    # Draw face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw emotion label and confidence
    label = f"{emotion}: {confidence:.2f}"
    cv2.putText(frame, label, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw prediction bars if provided
    if all_predictions is not None:
        draw_prediction_bars(frame, all_predictions, x, y + h + 10)

def draw_prediction_bars(frame, predictions, x, y, bar_width=200, bar_height=15):
    """Draw prediction probability bars"""
    spacing = 5
    
    for i, (emotion, prob) in enumerate(predictions.items()):
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        
        # Bar background
        bg_y = y + i * (bar_height + spacing)
        cv2.rectangle(frame, (x, bg_y), 
                     (x + bar_width, bg_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Bar fill
        fill_width = int(bar_width * prob)
        cv2.rectangle(frame, (x, bg_y), 
                     (x + fill_width, bg_y + bar_height), 
                     color, -1)
        
        # Label text
        label = f"{emotion}: {prob:.2f}"
        cv2.putText(frame, label, (x + 5, bg_y + bar_height - 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)