import tensorflow as tf

print("📦 Loading SavedModel...")
converter = tf.lite.TFLiteConverter.from_saved_model("../models/isl_saved_model")
tflite_model = converter.convert()

print("💾 Saving isl_model.tflite...")
with open("../models/isl_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete.")
