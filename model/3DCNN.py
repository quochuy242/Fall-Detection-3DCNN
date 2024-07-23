from keras import layers, Model, optimizers, callbacks, metrics
from utils import LABEL2INDEX, INDEX2LABEL, LABELS, set_seed, IMAGE_SIZE


def build_3DCNN(
    img_height: int = IMAGE_SIZE[0], img_width: int = IMAGE_SIZE[1]
) -> Model:
    set_seed()
    inputs = layers.Input((img_height, img_width, 1))
    x = layers.Rescaling(scale=1.0 / 255, offset=0.0)(inputs)
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Conv3D(filters=256, kernel_size=(3, 3, 3), activation="relu")(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    outputs = layers.Dense(len(LABELS), activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs, name="3DCNN")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
