from pathlib import Path
from time import time

from loguru import logger

from data import (
    create_3DCNN_dataset,
    create_df,
    train_val_test_split,
    count_elements,
    load_dataset,
)

if __name__ == "__main__":
    logger.add(
        "logs/dataset.log",
        colorize=False,
        backtrace=True,
        diagnose=True,
    )

    data_df = create_df(
        dir=Path("ViFam"), destination=Path("Dataset/csv"), return_df=True
    )

    train_df, val_df, test_df = train_val_test_split(
        df=data_df, destination=Path("Dataset/csv"), random_state=242
    )

    start = time()
    create_3DCNN_dataset(
        list_df=[train_df, val_df, test_df],
        destination=Path("Dataset/tfds"),
        ds_return=False,
        ds_save=True,
    )
    logger.info(f"Time taken: {time() - start}")

    train_ds, val_ds, test_ds = (
        load_dataset("Training"),
        load_dataset("Validation"),
        load_dataset("Testing"),
    )
    logger.info(count_elements(train_ds, val_ds, test_ds))
