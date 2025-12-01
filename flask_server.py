from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

print("✅ Starting Flask server...")
print("📦 Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path="isl_model.tflite")
interpreter.allocate_tensors()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,           # ✅ Enables tracking across frames
    max_num_hands=1,                   # ✅ More stable for single hand
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

class_labels = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

app = Flask(__name__)

def detect_hands(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    print("🧪 Running hand detection...")

    if not results.multi_hand_landmarks:
        print("🚫 No hand detected.")
        return [], []

    h, w, _ = img.shape
    boxes = []
    crops = []

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

    return crops, boxes

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    return img

def decode_prediction(pred):
    return class_labels[np.argmax(pred)]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        file_bytes = file.read()
        img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cv2.imwrite("debug_input.jpg", img)

        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        crops, boxes = detect_hands(img)
        if not crops:
            return jsonify({'gesture': '', 'box': [], 'confidence': []})

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        best_label = ''
        best_box = []
        best_conf = []
        best_score = -1

        for crop, box in zip(crops, boxes):
            if crop.size == 0:
                continue

            processed = preprocess(crop)
            if processed.shape != (64, 64, 3):
                continue

            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(processed, axis=0).astype(np.float32))
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            label = decode_prediction(prediction)
            score = np.max(prediction)

            print("🔮 Prediction vector:", prediction)
            print("✅ Predicted label:", label, "Confidence:", score)

            if score > best_score:
                best_score = score
                best_label = label
                best_box = box
                best_conf = prediction[0].tolist()

        os.remove("debug_input.jpg")  # Optional cleanup

        return jsonify({
            'gesture': best_label,
            'box': best_box,
            'confidence': best_conf
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
