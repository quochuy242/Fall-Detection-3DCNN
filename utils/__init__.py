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

## Image size
IMAGE_SIZE: tuple = (32, 32)
DEPTH: int = 1


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


def set_seed(seed: int = 242) -> None:
    """Set seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)
    return None
