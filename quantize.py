import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open("isl_model_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("✅ Quantized model saved as isl_model_quant.tflite")

