"""General utility functions"""
import os
import json
import logging
import random
import numpy as np
import pandas as pd
import datetime
from sklearn import metrics
import time
from PIL import Image
import cv2
from collections import defaultdict, deque, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


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


def plot_samples(
    df: pd.DataFrame,
    root_dir: str,
    rows: int = 3,
    cols: int = 5,
    figsize: tuple = (15, 10),
):
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
        vmin=0.2,
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
            color=("green" if preds[idx] == labels[idx] else "red"),
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
def get_albu_train_transforms(height, width, means, stds):
    """Apply training transformations from albumentation library"""
    trn_transform = A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.Perspective(p=0.5),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(),
        ]
    )
    return trn_transform


def get_albu_val_transforms(height, width, means, stds):
    """Apply val transformations from albumentation library"""
    val_transform = A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST),
            A.Normalize(mean=means, std=stds),
            ToTensorV2(),
        ]
    )
    return val_transform


# https://github.com/pytorch/vision/blob/main/references/classification/presets.py
def get_pt_train_transforms(
    height,
    width,
    means,
    stds,
    random_erase_prob,
    interpolation=InterpolationMode.BILINEAR,
    auto_augment_policy=None,
):
    """Apply training transformations from torchvision library"""
    trn_transform = []
    trn_transform = [
        transforms.RandomResizedCrop(
            (height, width), interpolation=interpolation
        ),
        transforms.RandomHorizontalFlip(0.5),
    ]
    # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#trivialaugment
    if auto_augment_policy is not None:
        if auto_augment_policy == "ra":
            trn_transform.append(
                autoaugment.RandAugment(interpolation=interpolation)
            )
        elif auto_augment_policy == "ta_wide":
            trn_transform.append(
                autoaugment.TrivialAugmentWide(interpolation=interpolation)
            )
        elif auto_augment_policy == "auto":
            aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
            trn_transform.append(
                autoaugment.AutoAugment(
                    policy=aa_policy, interpolation=interpolation
                )
            )
    trn_transform.extend(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=means, std=stds),
        ]
    )
    # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#random-erasing
    if random_erase_prob > 0:
        trn_transform.append(transforms.RandomErasing(p=random_erase_prob))
    trn_transform = transforms.Compose(trn_transform)
    return trn_transform


def get_pt_val_transforms(
    height, width, means, stds, interpolation=InterpolationMode.BILINEAR
):
    """Apply val transformations from torchvision library"""
    val_transform = transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=interpolation),
            # transforms.CenterCrop(crop_height, crop_width),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=means, std=stds),
        ]
    )
    return val_transform


def set_global_seeds(seed: int):
    # Ensure deterministic behavior : https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


# based on : https://github.com/pytorch/vision/blob/8dcb5b810d85bd42edf73280db1ece38c487004c/references/classification/utils.py


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16
    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(
                        s, "cpu"
                    )
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg)

    def update_parameters(self, model):
        for p_swa, p_model in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(
                    self.avg_fn(
                        p_swa.detach(), p_model_, self.n_averaged.to(device)
                    )
                )
        self.n_averaged += 1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
