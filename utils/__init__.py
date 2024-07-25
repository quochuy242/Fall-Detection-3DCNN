import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf

# Constants
## Classname
LABELS = [
    "ForwardFall",
    "BackwardFall",
    "LeftFall",
    "RightFall",
    "GetDown",
    "SitDown",
    "Walk",
]

## Dictionary for converting between label and index
LABEL2INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX2LABEL = {index: label for index, label in enumerate(LABELS)}

## Information of dataset
IMAGE_SIZE: tuple = (32, 32)
DEPTH: int = 36
OVERLAP = 3


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
