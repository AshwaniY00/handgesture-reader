import tensorflow as tf
import cv2
import numpy as np

# 1️⃣ Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/home/sunny/Desktop/jobProtal/java/handgesture-reader/isl_model.tflite")
interpreter.allocate_tensors()

# 2️⃣ Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded.")
print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])  # should be [1, 36] or [1, 37]
# 3️⃣ Load and preprocess a test image
img = cv2.imread("/home/sunny/Desktop/jobProtal/java/handgesture-reader/test_A.jpg")
img = cv2.resize(img, (64, 64))          # resize to match model input
img = img.astype(np.float32) / 255.0     # normalize
img = np.expand_dims(img, axis=0)        # add batch dimension

# 4️⃣ Set the image as input to the interpreter
interpreter.set_tensor(input_details[0]['index'], img)

# 5️⃣ Run inference
interpreter.invoke()

# 6️⃣ Get prediction
prediction = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(prediction)

# 7️⃣ Map index to actual label
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
print("Predicted Label:", labels[predicted_class])
