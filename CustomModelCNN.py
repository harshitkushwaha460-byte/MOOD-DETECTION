from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)


def build_model():
    model = Sequential()

    # Block 1
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(48, 48, 1)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(
        Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(7, activation="softmax"))

    return model