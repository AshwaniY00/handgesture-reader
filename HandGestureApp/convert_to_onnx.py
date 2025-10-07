import tensorflow as tf
import tf2onnx

# Load your Keras model
model = tf.keras.models.load_model("../models/isl_model.h5")

# Define input spec: batch size is flexible (None), image size is 64x64 with 3 channels
spec = (tf.TensorSpec((None, 64, 64, 3), tf.float32, name="input"),)

# Convert and save as ONNX
output_path = "../models/isl_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

print(f"✅ Model converted and saved to {output_path}")
