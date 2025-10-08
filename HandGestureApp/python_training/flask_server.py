from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

print("✅ Starting Flask server...")
print("📦 Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path="/home/sunny/Desktop/jobProtal/java/handgesture-reader/models/isl_model.tflite")
interpreter.allocate_tensors()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # ✅ Real-time tracking
    max_num_hands=1,
    min_detection_confidence=0.5,  # ✅ Helps detect hand better
    min_tracking_confidence=0.5    # ✅ Enables tracking
)

app = Flask(__name__)

def detect_hand(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    print("🧪 Running hand detection...")

    if not results.multi_hand_landmarks:
        print("🚫 No hand detected.")
        return None, []

    h, w, _ = img.shape
    hand_landmarks = results.multi_hand_landmarks[0]
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
    print("📦 Flask box:", box)

    cropped = img[y_min:y_max, x_min:x_max]
    return cropped, box

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return img

def decode_prediction(pred):
    return chr(np.argmax(pred) + ord('A'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        file_bytes = file.read()
        img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        hand_crop, box = detect_hand(img)
        if hand_crop is None:
            return jsonify({'gesture': 'No hand detected', 'box': []})

        processed = preprocess(hand_crop)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(processed, axis=0).astype(np.float32))
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        label = decode_prediction(prediction)
        return jsonify({'gesture': label, 'box': box})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
