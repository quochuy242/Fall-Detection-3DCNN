import datetime as dt
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import (
    Model,
    callbacks,
    losses,
    metrics,
    optimizers,
    regularizers,
)
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from data import one_hot

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
EPOCHS = 500


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


def lauching_tensorboard(log_dir: str) -> None:
    """
    Lauching tensorboard by running script on shell

    Args:
        log_dir (str): log directory for tensorboard"""
    logger.info(f"Launching tensorboard at {log_dir}")
    os.system(f"tensorboard --logdir={log_dir}")
    return None


def show_confusion_matrix(
    model: Model, test_ds: tf.data.Dataset, save_path: str
) -> None:
    """
    Show confusion matrix

    Args:
        model (Model): model object want to show confusion matrix
        test_ds (tf.data.Dataset): testing dataset, should be in form of (batch, image, label)
        save_path (str): path to save confusion matrix
    """
    y_pred = model.predict(test_ds)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.array(map(one_hot, y_pred))
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap="Blues")
    plt.savefig(f"{save_path}/confusion_matrix/{dt.datetime.now()}.png")
