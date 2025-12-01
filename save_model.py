import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your existing .h5 model
model = load_model("../models/isl_model.h5")

# Save it in SavedModel format
tf.saved_model.save(model, "../models/isl_saved_model")

print("✅ Model saved in SavedModel format.")
