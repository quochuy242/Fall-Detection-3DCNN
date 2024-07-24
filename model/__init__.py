from keras import (
    layers,
    Model,
    optimizers,
    callbacks,
    metrics,
    losses,
    regularizers,
)
from utils import set_seed, IMAGE_SIZE, DEPTH
from typing import List
from loguru import logger
import os

# Typings
Metrics = List[metrics.Metric]
Optimizer = optimizers.Optimizer
LossFunction = losses.Loss
Regularizer = regularizers.Regularizer
Callbacks = List[callbacks.Callback]

# Constants
## Classname
LABELS = ["ForwardFall", "BackwardFall", "LeftFall", "RightFall", "GetDown", "SitDown"]

## Dictionary for converting between label and index
LABEL2INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX2LABEL = {index: label for index, label in enumerate(LABELS)}

## Training parameters
BATCH_SIZE = 64
EPOCHS = 100


def compile(
    optimizer: Optimizer, loss: LossFunction, metrics: Metrics, model: Model
) -> Model:
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def summary(model: Model, log: bool = False) -> None:
    if log:
        logger.info(model.summary())
    else:
        model.summary()


def lauching_tensorboard(log_dir: str) -> callbacks.TensorBoard:
    logger.info(f"Launching tensorboard at {log_dir}")
    os.system(f"tensorboard --logdir={log_dir}")
