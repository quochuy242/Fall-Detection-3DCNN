from utils import INDEX2LABEL, URL_WEIGHTS, OUTPUT_WEIGHT_DOWNLOAD
from data import RealVideo
from keras import models, Model
import gdown
import os
import numpy as np
from typing import List
from tqdm import trange
from train_3DCNN_model import create_3DCNN_model


def load_model_from_url(
    model: Model = None,
    url: str = URL_WEIGHTS,
    output_path: str = OUTPUT_WEIGHT_DOWNLOAD,
) -> models.Model:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)
    model.load_weights(output_path)
    return model


def load_video(dir: str) -> RealVideo:
    return RealVideo(dir)


def fall(predictions: List[str]) -> str:
    for pred in predictions:
        if "fall" in pred.lower():
            return "FALL"
    return "NOT FALL"


if __name__ == "__main__":
    # Load model
    model = create_3DCNN_model()
    model = load_model_from_url(model=model)
    model.summary()

    # Load video
    videos = load_video(dir="demo")

    # Predict
    class_names = []
    for index in trange(len(videos), desc="Predicting"):
        pred = model.predict(videos.loading(index=index), verbose=0)
        pred = np.argmax(pred, axis=1)
        class_names.append([INDEX2LABEL[i] for i in pred])

    # Show result
    result = [fall(class_name) for class_name in class_names]
    print(f"Prediction: {result}")
