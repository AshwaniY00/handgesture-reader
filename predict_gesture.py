import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
interpreter = tf.lite.Interpreter(model_path="isl_model.tflite")
interpreter.allocate_tensors()

# Load and prepare the image
img = Image.open("test.jpg").resize((64, 64)).convert("RGB")
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Run prediction
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Get predicted class index
predicted_index = np.argmax(output)

# Map index to label (copy this from your training output)
class_labels = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'Alphabet', 'B', 'C', 'Conversation',
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

print("Predicted gesture:", class_labels[predicted_index])
