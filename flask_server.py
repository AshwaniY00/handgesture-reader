from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

print("✅ Starting Flask server...")
print("📦 Loading TFLite model...")

# Ensure captures directory exists
os.makedirs("captures", exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="isl_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Gesture labels (adjust to match your dataset)
class_labels = [
    '1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
]

CONFIDENCE_THRESHOLD = 0.07

app = Flask(__name__)
CORS(app)

# ----- Option 1: Homepage & health -----
@app.route('/', methods=['GET'])
def home():
    return "Gesture Recognition API is running! Use POST /predict with form-data key 'image'."

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def detect_hands(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    print("🧪 Running hand detection...")

    if not results.multi_hand_landmarks:
        print("🚫 No hand detected.")
        return [], []

    h, w, _ = img.shape
    boxes, crops = [], []

    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        pad = 20
        x_min = max(x_min - pad, 0)
        y_min = max(y_min - pad, 0)
        x_max = min(x_max + pad, w)
        y_max = min(y_max + pad, h)

        box = [x_min, y_min, x_max, y_max]
        cropped = img[y_min:y_max, x_min:x_max]
        boxes.append(box)
        crops.append(cropped)
        print("📦 Detected box:", box)

    return boxes, crops

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    return img

def decode_prediction(pred):
    return class_labels[np.argmax(pred)]

# ----- Option 2: Prediction endpoint -----
@app.route('/predict', methods=['POST'])
def predict():
    print("✅ Received request at /predict")   # Debug print added here
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        file_bytes = file.read()
        img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        boxes, crops = detect_hands(img)

        if not crops:
            return jsonify({
                'gesture': '',
                'confidence': 0,
                'box': None,
                'top_predictions': []
            })

        best_label = ''
        best_box = []
        best_score = -1
        top_predictions = []

        for crop, box in zip(crops, boxes):
            if crop.size == 0:
                continue

            processed = preprocess(crop)
            if processed.shape != (64, 64, 3):
                continue

            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(processed, axis=0).astype(np.float32))
            interpreter.invoke()
            raw_pred = interpreter.get_tensor(output_details[0]['index'])[0]
            confidence_vector = softmax(raw_pred)

            label = decode_prediction(confidence_vector)
            score = float(np.max(confidence_vector))

            top_indices = np.argsort(confidence_vector)[::-1][:3]
            top_predictions = [(class_labels[i], float(confidence_vector[i])) for i in top_indices]

            if score > best_score:
                best_score = score
                best_label = label
                best_box = box
                top_predictions = [(class_labels[i], float(confidence_vector[i])) for i in top_indices]

                if best_score > 0.8:
                    cv2.imwrite(f"captures/{best_label}_{int(best_score*100)}.png", crop)
                    print(f"💾 Saved crop: {best_label}_{int(best_score*100)}.png")

        if best_score < CONFIDENCE_THRESHOLD:
            return jsonify({
                'gesture': '',
                'confidence': 0,
                'box': None,
                'top_predictions': []
            })

        response = {
            'gesture': best_label,
            'confidence': best_score,
            'box': best_box,
            'top_predictions': top_predictions
        }

        print("📤 Final response:", response)
        return jsonify(response)

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Local dev only; on Render, gunicorn starts the app
    app.run(port=5001)
