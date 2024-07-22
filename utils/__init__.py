import cv2
from pathlib import Path
from typing import List

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
