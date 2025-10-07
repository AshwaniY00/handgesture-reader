import tensorflow as tf
print("📦 Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path="../models/isl_model.tflite")
interpreter.allocate_tensors()

print("✅ Model loaded successfully.")
