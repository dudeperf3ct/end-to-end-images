"""General utility functions"""
import os
import json
import logging
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from PIL import Image
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self) -> str:
        return str(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def datasets_to_df(ds_path: str):
    """
    Convert dataset folder to pandas dataframe format

    Args:
        ds_path (string): Path to dataset

    Returns:
        pd.DataFrame : A pandas dataframe containing paths to dataset and labels.
    """
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dataset directory not found: {ds_path}")
    raise NotImplementedError("Implement this method")
    # return pd.DataFrame(data, columns=["file", "label"]) --> return same dataframe to be consistent

def plot_hist(history: dict):
    """
    Plot training and validation accuracy and losses

    Args:
        history: Dict containing training loss, acc and val loss, acc
    """
    # summarize history for accuracy
    plt.plot(history["train_acc"])
    plt.plot(history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()
    # summarize history for loss
    plt.plot(history["train_loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


def img_display(img, means, stds):
    """
    Convert normalized image to display unnormalized image
    """
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    # unnormalize
    img = img * np.array(stds) + np.array(means)
    return img


def plot_samples(df: pd.DataFrame, root_dir: str, rows: int = 3, cols: int = 5, figsize: tuple = (15, 10)):
    """
    Plot random sample of 15 images in grid of rows x cols
    Args:
        df: Pandas dataframe with columns ["file", "labels"] containing image path and corresponding labels
        root_dir: Root directory if the path in dataframe are relative
        rows: Number of rows in grid
        cols: Number of columns in grid
        figsize: Tuple containing matplotlib figure size

    Returns: plot of matplotlib figure containing rows*cols samples

    """
    # get some random training images
    fig, axis = plt.subplots(rows, cols, figsize=figsize)
    samples = df.sample(rows * cols)
    images, labels = samples["file"].values, samples["label"].values
    # Viewing data examples used for training
    for i, ax in enumerate(axis.flat):
        img, lbl = images[i], labels[i]
        im = Image.open(os.path.join(root_dir, img))
        ax.imshow(im)
        ax.set(title=f"{lbl}")
    return fig


def plot_cm(true, preds, classes, figsize: tuple = (8, 6)):
    """Plot confusion matrix"""
    cm = metrics.confusion_matrix(true, preds)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0.2
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    return fig


def plot_predictions(model, test_loader, device, classes: list, means, stds):
    """Plot predictions for 20 examples in a grid of 2x10"""
    model.eval()
    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.tolist()
    with torch.no_grad():
        # get sample outputs
        output = model(images)
    probs = F.softmax(output)
    probs_tensor, _ = torch.max(probs, 1)
    probs_tensor = probs_tensor.cpu().numpy() * 100
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    print("Grid for 20 examples: Pred (Label)")
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(img_display(images[idx].cpu(), means, stds))
        ax.set_title(
            "{} ({}) {:.2f}".format(
                classes[preds[idx]], classes[labels[idx]], probs_tensor[idx]
            ),
            color=("green" if preds[idx] == labels[idx] else "red")
        )
    return fig


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


# https://github.com/ultralytics/yolov5/blob/master/utils/general.py
def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


# strong augmentations
def get_train_transforms(height, width, means, stds):
    """
    Apply training transformations from albumentation library
    """
    trn_transform = A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),
            A.Normalize(mean=means, std=stds),
            ToTensorV2()
        ]
    )
    return trn_transform


def get_val_transforms(height, width, means, stds):
    """
    Apply val transformations from albumentation library
    """
    val_transform = A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST),
            A.Normalize(mean=means, std=stds),
            ToTensorV2()
        ]
    )
    return val_transform


def set_global_seeds(seed: int):
    # Ensure deterministic behavior : https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
