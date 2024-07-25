from keras import layers, Model, regularizers
from utils import LABELS, set_seed, IMAGE_SIZE, DEPTH
from typing import Callable


def add_conv_block(
    input,
    filters: int,
    kernel_size: tuple = (3, 3, 3),
    pool_size: tuple = (2, 2, 2),
    activation: str = "relu",
):
    x = layers.Conv3D(filters=filters, kernel_size=kernel_size, activation=activation)(
        input
    )
    x = layers.MaxPool3D(pool_size=pool_size)(x)
    x = layers.BatchNormalization()(x)
    return x


def add_dense_block(
    input, units: int, activation: str = "relu", dropout_rate: float = 0.2
):
    x = layers.Dense(units=units, activation=activation)(input)
    x = layers.Dropout(rate=dropout_rate)(x)
    return x


def build(
    num_conv_blocks: int,
    num_dense_blocks: int,
    seed: int,
    activation: Callable,
    initial_filters: int = 32,
    initial_dense_units: int = 1024,
) -> Model:
    # Set seed for any random operations
    set_seed(seed)

    # Input layer
    input_layer = layers.Input(shape=(DEPTH, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)

    # Add convolutional blocks
    for i in range(num_conv_blocks):
        x = add_conv_block(x, filters=int(initial_filters * (2**i)))

    # Add dense blocks
    x = layers.Flatten()(x)
    for i in range(num_dense_blocks):
        x = add_dense_block(x, units=int(initial_dense_units * (2 ** (-i))))

    # Output layer
    output = layers.Dense(
        len(LABELS),
        activation=activation,
        kernel_regularizer=regularizers.l1(0.004),
        activity_regularizer=regularizers.l2(0.004),
    )(x)

    # Create model
    model = Model(inputs=input_layer, outputs=output)
    return model
