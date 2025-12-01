import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="isl_model.tflite")
interpreter.allocate_tensors()


# Show output shape (number of gesture classes)
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded.")
print("Output shape:", output_details[0]['shape'])  # e.g., [1, 37]

