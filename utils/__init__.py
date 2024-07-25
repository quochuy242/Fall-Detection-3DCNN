import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf
import yaml

# Load config and params from yaml file
with open("config.yaml", "r") as stream:
    config = yaml.safe_load(stream)
with open("train_params.yaml", "r") as stream:
    train_params = yaml.safe_load(stream)
with open("test_params.yaml", "r") as stream:
    test_params = yaml.safe_load(stream)

# Global variables
LABELS = config["labels"]
IMAGE_SIZE = (config["image_width"], config["image_height"])
DEPTH = config["depth"]
OVERLAP = config["overlap"]
CHANNEL = config["image_channel"]
BATCH_SIZE = train_params["batch_size"]
EPOCHS = train_params["epochs"]
OPTIMIZER = train_params["optimizer"]
URL_WEIGHTS = test_params["url_weights"]
OUTPUT_WEIGHT_DOWNLOAD = test_params["output_weight_download"]


# Dictionary for converting between label and index
LABEL2INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX2LABEL = {index: label for index, label in enumerate(LABELS)}


# Useful function
def count_frames(video: cv2.VideoCapture) -> int:
    """Count the number of frames in a video"""
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def get_all_paths(path: Path) -> List[Path]:
    """Get all video files in a directory"""
    return list(path.glob("*.mp4")) + list(path.glob("*.MOV"))


def get_label_from_path(path: Path) -> str:
    """Get label from path"""
    return path.parent.name


def get_fps_from_video(path: Path) -> int:
    """Get fps from video path"""
    return int(np.ceil(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS)))


def get_resolution_from_video(path: Path) -> tuple:
    """Get size from video path"""
    return (
        int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def set_seed(seed: int = 242) -> None:
    """Set seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    return None
