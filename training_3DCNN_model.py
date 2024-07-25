import datetime as dt

from keras import activations, callbacks, losses, metrics, optimizers, utils
from loguru import logger

from data import (
    load_dataset,
    prepare_dataset,
)
from model import (
    CNN_3D,
    EPOCHS,
    compile,
    lauching_tensorboard,
    show_confusion_matrix,
    summary,
)

if __name__ == "__main__":
    # Add log directory to log training progress
    logger.add(
        "logs/fit/training.log",
        colorize=False,
        backtrace=True,
        diagnose=True,
    )

    # Load dataset
    logger.info("Loading dataset...")
    train_ds, val_ds, test_ds = (
        load_dataset("Training"),
        load_dataset("Validation"),
        load_dataset("Testing"),
    )

    # Prepare dataset
    logger.info("Preparing dataset...")
    try:
        train_ds, val_ds, test_ds = (
            prepare_dataset(train_ds),
            prepare_dataset(val_ds),
            prepare_dataset(test_ds),
        )
    except Exception as e:
        logger.error(e)

    # Build 3DCNN model
    logger.info("Building 3DCNN model...")
    cnn_3d = CNN_3D.build(
        num_conv_blocks=2, num_dense_blocks=2, seed=242, activation=activations.softmax
    )

    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    best_checkpoint = callbacks.ModelCheckpoint(
        f"checkpoint/{dt.date.today()}/best.keras",
        save_best_only=True,
        monitor="val_f1_score",
    )
    last_checkpoint = callbacks.ModelCheckpoint(
        f"checkpoint/{dt.date.today()}/last.keras",
    )
    tensorboard_cb = callbacks.TensorBoard(
        log_dir=f"logs/{dt.date.today()}", histogram_freq=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, min_lr=5e-5
    )

    # Compile 3DCNN model
    cnn_3d = compile(
        optimizer=optimizers.Adadelta(learning_rate=5e-3, epsilon=1e-08),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.AUC(),
            metrics.F1Score(average="weighted"),
            metrics.Recall(),
            metrics.Precision(),
        ],
        model=cnn_3d,
    )
    summary(model=cnn_3d, log=True)

    # Plotting the architectural graph of the model
    utils.plot_model(
        cnn_3d, to_file="visualize/architecture_3DCNN.png", show_shapes=True, dpi=60
    )

    # Lauching tensorboard
    log_dir = f"logs/{dt.now().strftime('%Y%m%d-%H%M%S')}"
    lauching_tensorboard(log_dir=log_dir)

    # Training 3DCNN model
    logger.info("Training 3DCNN model...")

    cnn_3d.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tensorboard_cb,
            early_stopping,
            best_checkpoint,
            last_checkpoint,
            reduce_lr,
        ],
    )

    # Evaluate 3DCNN model
    logger.info("Evaluating 3DCNN model...")
    logger.info(cnn_3d.evaluate(test_ds, return_dict=True))

    # Plot confusion matrix
    logger.info("Plotting confusion matrix...")
    show_confusion_matrix(cnn_3d, test_ds, "visualize/confusion_matrix")

    logger.info("All done!")
