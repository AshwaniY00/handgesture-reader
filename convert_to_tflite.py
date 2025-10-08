import tensorflow as tf

print("📦 Loading .h5 model...")
from keras.models import load_model
model = load_model("../models/isl_model.h5")
print("🔄 Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("💾 Saving isl_model.tflite...")
with open("../models/isl_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete.")
