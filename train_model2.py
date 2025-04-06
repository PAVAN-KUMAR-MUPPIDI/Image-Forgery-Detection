import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Define image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Ensure dataset directories exist
TRAIN_DIR = "dataset_training"
VALIDATION_DIR = "dataset_validation"

if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError("Dataset directories not found! Check paths.")

# Data Augmentation & Normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", subset="training"
)
validation_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", subset="validation"
)

# CNN Model
model = Sequential([
    Input(shape=(128, 128, 3)),  # Fix input shape issue
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks for best model saving & early stopping
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("best_image_forgery_model.keras", save_best_only=True)  # ❌ Removed `save_format="keras"`
]

# Train model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=callbacks
)

# Save final model
model.save("image_forgery_detection_model.keras")  # Ensures final model is saved properly

# Plot accuracy/loss
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Training Performance")
plt.show()