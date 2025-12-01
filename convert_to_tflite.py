import tensorflow as tf

print("📦 Converting SavedModel to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model("../models/isl_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("../models/isl_model.tflite", "wb") as f:

    f.write(tflite_model)

print("✅ TFLite model saved as isl_model.tflite")
