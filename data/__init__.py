import os
import pandas as pd
import numpy as np
import cv2
from utils import IMAGE_SIZE, LABELS, LABEL2INDEX, INDEX2LABEL
from utils import get_label_from_path, get_all_paths, count_frames
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple


def create_df(dir: Path, destination: Path = Path("data")) -> pd.DataFrame:
    """Create dataframe to create dataset from directory

    Args:
        dir (Path): dataset's directory
        destination (Path): path to save dataframe, Defaults to Path("data")

    Returns:
        pd.DataFrame: Dataframe having two columns: path and label
    """
    df = pd.DataFrame(columns=["path", "label"])
    for label in tqdm(LABELS, desc=f"Creating dataframe"):
        data_dir = dir / label
        files: list[Path] = get_all_paths(data_dir)
        file_dicts = {
            "path": files,
            "label": [get_label_from_path(file) for file in files],
        }
        df = pd.concat([df, pd.DataFrame(file_dicts)], axis=0)
    df.to_csv(str(destination / f"data.csv"), index=False)
    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    destination: Path = Path("data"),
) -> Tuple[pd.DataFrame]:
    """Split dataframe into train, validation and test dataframes

    Args:
        df (pd.DataFrame): Dataframe having two columns: path and label
        train_ratio (float, optional): Ratio of train data. Defaults to 0.6.
        val_ratio (float, optional): Ratio of validation data. Defaults to 0.2.
    """
    test_ratio = 1 - train_ratio - val_ratio
    train_df, val_df, test_df = (
        df.sample(n=int(df.shape[0] * train_ratio), random_state=242),
        df.sample(n=int(df.shape[0] * val_ratio), random_state=242),
        df.sample(n=int(df.shape[0] * test_ratio), random_state=242),
    )
    train_df.to_csv(str(destination / "train.csv"), index=False)
    val_df.to_csv(str(destination / "val.csv"), index=False)
    test_df.to_csv(str(destination / "test.csv"), index=False)

    return train_df, val_df, test_df


def create_overlap_video(
    video_path: Path, clip_length: int = 36, overlap: int = 3
) -> np.ndarray:
    """Create overlapping video. Each video frame consists of 36 consecutive frames with overlap

    Args:
        video_path (str): Path to video
        clip_length (int, optional): Length of clip. Defaults to 36.
        overlap (int, optional): Overlap between two consecutive frames. Defaults to 3.
    """

    frames = []  # List to contain all frames

    vid = cv2.VideoCapture(video_path)
    num_frame = count_frames(vid)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        success, frame = vid.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if width != IMAGE_SIZE[0] or height != IMAGE_SIZE[1]:
            frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        
        frames.append(frame)
    vid.release()

    for start in range(0, num_frame - clip_length + 1, overlap):
        clip = frames[start : start + clip_length]
        if len(clip) == 36:
            video


def create_3DCNN_dataset(
    list_df: List[pd.DataFrame], destination: Path = Path("data")
) -> Tuple[np.ndarray, ...]:
    """Create overlap video dataset. Each frame of video consists of 36 consecutive frames

    Args:
        list_df (List[pd.DataFrame]): List of dataframe
        destination (Path, optional): Path to save dataset. Defaults to Path("data").
    """
