import tensorflow as tf
import cv2
import numpy as np
import os

# 1️⃣ Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/sunny/Desktop/jobProtal/java/handgesture-reader/isl_model.tflite")
interpreter.allocate_tensors()

# 2️⃣ Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded.")
print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# 3️⃣ Define labels (adjust if your dataset has fewer than 36 classes)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# 4️⃣ Folder path
folder_path = "/home/sunny/Desktop/jobProtal/java/handgesture-reader/"

# 5️⃣ Only test specific files
test_files = ["test_A.jpg", "test_B.jpg"]

for filename in test_files:
    file_path = os.path.join(folder_path, filename)

    # Load and preprocess image
    img = cv2.imread(file_path)
    if img is None:
        print(f"⚠️ Could not read {filename}")
        continue

    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction)

    # Print result
    if predicted_class < len(labels):
        print(f"{filename} ➝ Predicted Label: {labels[predicted_class]}")
    else:
        print(f"{filename} ➝ Predicted class index {predicted_class} (no matching label)")
