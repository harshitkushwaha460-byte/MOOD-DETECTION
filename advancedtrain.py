import os

import tensorflow as tf

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

# -----------------------
# Configuration
# -----------------------

IMG_SIZE = (224,224)

BATCH_SIZE = 16

EPOCHS = 10

TRAIN_DIR = "data/train"

VAL_DIR = "data/val"

# -----------------------
# Data Generator
# -----------------------

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"
)

# -----------------------
# Load EfficientNetB0
# -----------------------

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

# -----------------------
# Custom Classifier
# -----------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(
    256,
    activation="relu"
)(x)

x = Dropout(0.5)(x)

output = Dense(
    7,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# -----------------------
# Compile
# -----------------------

model.compile(

    optimizer=Adam(1e-4),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

# -----------------------
# Callbacks
# -----------------------

os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(

    "models/best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)

early_stop = EarlyStopping(

    monitor="val_accuracy",

    patience=5,

    restore_best_weights=True,

    mode="max",

    verbose=1

)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# -----------------------
# Train
# -----------------------

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=EPOCHS,

    callbacks=[checkpoint,early_stop,reduce_lr]

)

# -----------------------
# Fine Tuning
# -----------------------

print("\nStarting Fine-Tuning...\n")

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(

    optimizer=Adam(1e-5),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

history_fine = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=10,

    callbacks=[checkpoint,early_stop]

)

print("Training Completed Successfully!")