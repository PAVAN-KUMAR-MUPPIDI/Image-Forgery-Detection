import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, 
                                     GlobalAveragePooling2D, Add)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Define constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_DIR = "dataset_training"
VALIDATION_DIR = "dataset_validation"

# Ensure dataset directories exist
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError("Dataset directories not found! Check paths.")

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", subset="training"
)
validation_generator = train_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", subset="validation"
)

# 🔹 Fixed Residual Model
def build_improved_model(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)

    # First block
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Residual Block 1 (Fixed Skip Connection)
    shortcut = Conv2D(64, (1, 1), padding="same")(x)  # Fix shape mismatch
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x = Add()([shortcut, x1])  # Now shapes match
    x = MaxPooling2D((2, 2))(x)

    # Residual Block 2
    shortcut = Conv2D(128, (1, 1), padding="same")(x)  # Fix shape mismatch
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x = Add()([shortcut, x2])
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    return model

# Build and compile model
model = build_improved_model()
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_image_forgery_model.keras", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)  # Adjusts learning rate dynamically
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final model
model.save("image_forgery_detection_model.keras")

# Plot training performance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs. Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs. Validation Loss")

plt.show()
