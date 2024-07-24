import os
import pickle
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from tqdm import tqdm

from utils import (
    DEPTH,
    IMAGE_SIZE,
    LABEL2INDEX,
    LABELS,
    OVERLAP,
    count_frames,
    get_all_paths,
    get_fps_from_video,
    get_label_from_path,
    get_resolution_from_video,
)


def create_df(
    dir: Path, destination: Path = Path("Dataset"), return_df: bool = False
) -> pd.DataFrame | None:
    """Create dataframe to create dataset from directory

    Args:
        dir (Path): dataset's directory
        destination (Path): path to save dataframe, Defaults to Path("data")
        return_df (bool): Whether to return the DataFrame, default is False
    Returns:
        pd.DataFrame: Dataframe having two columns: path and label
    """
    df = pd.DataFrame(columns=["path", "fps", "resolution", "num_frames", "label"])
    for label in tqdm(LABELS, desc="Creating dataframe"):
        data_dir = dir / label
        files: List[Path] = get_all_paths(data_dir)
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "path": [str(file) for file in files],
                        "fps": [get_fps_from_video(file) for file in files],
                        "resolution": [
                            get_resolution_from_video(file) for file in files
                        ],
                        "num_frames": [
                            count_frames(cv2.VideoCapture(str(file))) for file in files
                        ],
                        "label": [get_label_from_path(file) for file in files],
                    }
                ),
            ],
            axis=0,
        )

    os.makedirs(str(destination), exist_ok=True)
    df.to_csv(str(destination / "data.csv"), index=False)
    return df if return_df else None


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    destination: Path = Path("Dataset"),
    random_state: int = 242,
) -> Tuple[pd.DataFrame]:
    """Split dataframe into train, validation and test dataframes

    Args:
        df (pd.DataFrame): Dataframe having two columns: path and label
        train_ratio (float, optional): Ratio of train data. Defaults to 0.6.
        val_ratio (float, optional): Ratio of validation data. Defaults to 0.2.
    """
    test_ratio = 1 - train_ratio - val_ratio
    train_df, val_df, test_df = (
        df.sample(n=int(df.shape[0] * train_ratio), random_state=random_state),
        df.sample(n=int(df.shape[0] * val_ratio), random_state=random_state),
        df.sample(n=int(df.shape[0] * test_ratio), random_state=random_state),
    )

    os.makedirs(str(destination), exist_ok=True)
    train_df.to_csv(str(destination / "train.csv"), index=False)
    val_df.to_csv(str(destination / "val.csv"), index=False)
    test_df.to_csv(str(destination / "test.csv"), index=False)

    return train_df, val_df, test_df


# // FIX THAT: length of clips from .mp4 (69) is not same as one from .MOV (128)
def preprocess_all_frames(video_path: str) -> np.ndarray:
    """Create overlapping video. Each video frame consists of 36 consecutive frames with overlap

    Args:
        video_path (str): Path to video
        clip_length (int, optional): Length of clip. Defaults to 36.
        overlap (int, optional): Overlap between two consecutive frames. Defaults to 3.
    """

    frames = []  # List to contain all frames

    vid = cv2.VideoCapture(video_path)
    # num_frame = count_frames(vid)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    success, frame = vid.read()
    while success:
        # Preprocessing each frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if width != IMAGE_SIZE[0] or height != IMAGE_SIZE[1]:
            frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, axis=-1)

        # Append processed frame
        frames.append(frame)

        # Continue reading
        success, frame = vid.read()

    # Release video object
    vid.release()
    return np.array(frames)


def create_3DCNN_dataset(
    list_df: List[pd.DataFrame],
    clip_length: int = DEPTH,
    overlap: int = OVERLAP,
    destination: Path = Path("Dataset"),
    ds_return: bool = False,
    ds_save: bool = False,
) -> List[tf.data.Dataset] | None:
    """Create overlap video dataset. Each frame of video consists of 36 consecutive frames

    Args:
        list_df (List[pd.DataFrame]): List of dataframe
        destination (Path, optional): Path to save dataset. Defaults to Path("data").
    """

    if ds_return:
        ds_ = []

    logger.info("Creating overlapping video dataset...")
    for df, name in zip(list_df, ["Training", "Validation", "Testing"]):
        paths, labels = list(df["path"]), list(df["label"])
        labels = [LABEL2INDEX[label] for label in labels]

        # Create overlapping clips from all frames
        clip_frames = []
        clip_label = []
        for path, label in tqdm(
            zip(paths, labels), desc=f"{name} dataset", total=len(paths)
        ):
            frames = preprocess_all_frames(path)
            for i in range(0, len(frames) - clip_length + 1, clip_length - overlap):
                if len(frames[i : i + clip_length]) == clip_length:
                    clip_frames.append((frames[i : i + clip_length]))
                    clip_label.append(label)

        clip_frames = np.array(clip_frames)
        clip_label = np.array(clip_label)

        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((clip_frames, clip_label))
        dataset = dataset.shuffle(buffer_size=len(dataset) + 1)

        # Save dataset
        if ds_save:
            os.makedirs(str(destination), exist_ok=True)
            tf.data.Dataset.save(
                dataset, path=str(destination / name), compression="GZIP"
            )
            with open(str(destination / name / "element_spec"), "wb") as out_:
                pickle.dump(dataset.element_spec, out_)

        if ds_return:
            ds_.append(dataset)

    # Return dataset
    return ds_ if ds_return else None


def load_dataset(name: str, data_dir: Path = Path("Dataset/tfds")) -> tf.data.Dataset:
    """Load TFDS dataset

    Args:
        name (str): Name of dataset
    """

    with open(data_dir / name / "element_spec", "rb") as in_:
        element_spec = pickle.load(in_)

    return tf.data.Dataset.load(
        str(data_dir / name), element_spec=element_spec, compression="GZIP"
    )


def prepare_dataset(dataset: tf.data.Dataset, batch_size: int = 64) -> tf.data.Dataset:
    def one_hot(label):
        label = tf.cast(label, tf.int32)
        return tf.one_hot(label, len(LABELS))

    dataset = dataset.map(lambda x, y: (x, one_hot(y)))
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def count_elements(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> pd.DataFrame:
    _, train_count = np.unique(
        [output.numpy() for _, output in train_ds], return_counts=True
    )
    _, val_count = np.unique(
        [output.numpy() for _, output in val_ds], return_counts=True
    )
    _, test_count = np.unique(
        [output.numpy() for _, output in test_ds], return_counts=True
    )

    count_df = pd.DataFrame(
        {"Train": train_count, "Val": val_count, "Test": test_count}, index=LABEL2INDEX
    )
    count_df.loc["Total"] = {
        "Train": sum(train_count),
        "Val": sum(val_count),
        "Test": sum(test_count),
    }
    return count_df
