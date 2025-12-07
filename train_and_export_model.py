import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Parameters
input_shape = (64, 64, 3)
batch_size = 64
epochs = 50

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "ISL_Dataset/train",
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    "ISL_Dataset/val",
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)
import os
for folder in os.listdir("ISL_Dataset/train"):
    print(folder, len(os.listdir(os.path.join("ISL_Dataset/train", folder))))


# ✅ Use dataset-driven labels
labels = list(train_data.class_indices.keys())
num_classes = len(labels)
print("Class mapping:", train_data.class_indices)

# Improved CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_best.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-5
)

# Train
model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[lr_callback, checkpoint, early_stop]
)

# Save model
model.save("saved_model", save_format="tf")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("isl_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Improved model training and export complete.")

# ------------------ Testing ------------------

# Load the best saved model from checkpoints
model = load_model("checkpoints/model_best.h5")

# Test with an image
img = cv2.imread("test_A.jpg")        # replace with your image path
img = cv2.resize(img, (64, 64))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = np.argmax(prediction)

# ✅ Use dataset-driven labels
print("Predicted Label:", labels[predicted_class])

# Print top-3 predictions for debugging
probs = prediction[0]
top_indices = probs.argsort()[-3:][::-1]
print("Top predictions:")
for i in top_indices:
    print(f"{labels[i]}: {probs[i]:.2f}")
