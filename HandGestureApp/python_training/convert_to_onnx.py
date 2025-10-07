import tensorflow as tf
import keras2onnx
import onnx

# Load your trained model
model = tf.keras.models.load_model("isl_model.h5")

# Convert to ONNX format
onnx_model = keras2onnx.convert_keras(model, model.name)

# Save the ONNX model
onnx.save_model(onnx_model, "isl_model.onnx")
