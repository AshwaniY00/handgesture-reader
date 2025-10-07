import tensorflow as tf

model = tf.keras.models.load_model("/home/sunny/Desktop/models/isl_model.h5")
tf.saved_model.save(model, "/home/sunny/Desktop/models/isl_saved_model")
