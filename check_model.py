import tensorflow as tf

model = tf.keras.models.load_model("gesture_model_20251012_0954.h5")
print("✅ Model loaded successfully.")
print("Number of output classes:", model.output_shape[-1])
