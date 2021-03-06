import os
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import json
import warnings
from pprint import pprint
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import default_collate
import wandb

import transforms
from dataset import ImageClassifierDataset
from model import build_models
from trainer import Trainer
from inference.traceSaver import TraceSaver
from inference.validate_onnx_conversion import ValidateOnnx
from evaluate import test_model
from utils import (
    set_global_seeds,
    colorstr,
    Params,
    datasets_to_df,
    set_logger,
)
from utils import get_albu_train_transforms, get_albu_val_transforms
from utils import get_pt_val_transforms, get_pt_train_transforms


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ClassifierModel:
    def __init__(self, model_dir: str, experiment: str):
        """

        Args:
            model_dir: Path to folder containing `params.json` file
            experiment: Name of experiment used for logging in wandb
        """
        json_path = os.path.join(model_dir, "params.json")
        assert os.path.isfile(
            json_path
        ), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)
        log_path = os.path.join(model_dir, f'{model_dir.split("/")[-1]}.log')
        # set logger
        set_logger(log_path)
        logging.info(f"Creating a logging file at {log_path}")
        self.model_dir = model_dir
        # create a model folders to store models and read config for the experiments
        try:
            if os.path.isdir(model_dir):
                logging.info(f"Parent Directory {model_dir} exists!!")
            os.makedirs(os.path.join(model_dir, "models"), exist_ok=True)
            logging.info(
                f"Creating models directory at {os.path.join(model_dir, 'models')}"
            )
        except Exception as e:
            logging.info(e)
            logging.info("Error creating models directory. Check permissions!")
            quit()
        set_global_seeds(params.seed)
        # hyperparameters BEGIN
        # data config
        self.random_seed = params.seed
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#long-training
        self.epochs = params.epochs
        self.split_ratio = params.test_split_ratio
        self.num_classes = params.num_classes
        self.embed_size = params.embed_size
        self.input_channels = params.input_channels
        self.folds = params.folds
        # augmentations
        self.width = params.width
        self.height = params.height
        self.means = params.means
        self.stds = params.stds
        self.auto_augment_policy = params.auto_augment_policy
        self.random_erase_prob = params.random_erase_prob
        self.transform_type = params.transform_type
        # paths config
        self.train_root_dir = params.train_root_dir
        self.save_model_path = params.save_model_path
        self.trained_model_path = params.trained_model_path
        # model config
        self.model_name = params.model_name
        self.feature_extract = params.feature_extract
        self.use_pretrain = params.use_pretrain
        self.finetune_layer = params.finetune_layer
        # optimizer
        self.optimizer = params.optimizer.lower()
        self.lr = params.learning_rate
        self.momentum = params.momentum
        self.lr_step_size = params.lr_step_size
        self.lr_gamma = params.lr_gamma
        # regularization
        self.weight_decay = params.weight_decay
        self.norm_weight_decay = params.norm_weight_decay
        self.mixup_alpha = params.mixup_alpha
        self.cutmix_alpha = params.cutmix_alpha
        self.label_smoothing = params.label_smoothing
        # lr scheduler
        self.scheduler = params.scheduler.lower()
        self.lr_warmup_epochs = params.lr_warmup_epochs
        self.lr_warmup_method = params.lr_warmup_method.lower()
        self.lr_warmup_decay = params.lr_warmup_decay
        # visualize and logging
        self.use_wandb = params.use_wandb
        # converter
        self.convert_onnx = params.convert_onnx
        self.class_mapping = None
        self.classes = None
        if self.use_wandb:
            wandb.init(project=experiment)
            with open(json_path, "r") as f:
                param = json.load(f)
            wandb.config.update(param)
        # hyperparameters END
        logging.info("-" * 80)
        logging.info(colorstr("Hyperparameters: "))
        with open(json_path, "r") as f:
            param = json.load(f)
        pprint(param)
        logging.info("-" * 80)
        self.validate_params()
        self.net = self._init_model()

    def validate_params(self):
        if self.auto_augment_policy not in ["None", "ta-wide", "ra", "auto"]:
            raise ValueError(
                f"Autoaugment Policy supports choices : {['None', 'ta-wide', 'ra', 'auto']}"
            )
        if self.lr_warmup_method not in ["None", "linear", "constant"]:
            raise ValueError(
                f"LR Warmup methods supports choices : {['None', 'linear', 'constant']}"
            )
        if self.optimizer not in ["adamw", "adam", "radam", "sgd", "rmsprop"]:
            raise ValueError(
                f"Optimizers supports choices : {['adamw', 'adam', 'radam', 'sgd', 'rmsprop']}"
            )
        if self.scheduler not in [
            "steplr",
            "onecyclelr",
            "cosineannealinglr",
            "exponentiallr",
        ]:
            raise ValueError(
                f"Optimizers supports choices : {['steplr', 'onecyclelr', 'cosineannealinglr', 'exponentiallr']}"
            )
        if self.transform_type not in ["pt", "albu"]:
            raise ValueError(
                f"Augmentation type supports choices : {['pt', 'albu']}"
            )

    def _init_model(self):
        """Initialize model if .pth is found else create a new model"""
        if os.path.exists(self.save_model_path):
            logging.info(f"Found a model at {self.save_model_path}")
            classifier_model = torch.load(
                self.save_model_path, map_location=device
            )
        else:
            logging.info(f"Saved model not found at {self.save_model_path}")
            classifier_model = self._build_pretrain_model()
        return classifier_model

    # build a pretrained model with imagenet weights
    def _build_pretrain_model(self):
        """Create a pretrained model"""
        # force pretraining by setting best weights and finetune layers to defaults of pretraining
        classifier_model = build_models(
            model_name=self.model_name,
            num_classes=self.num_classes,
            in_channels=self.input_channels,
            embedding_size=self.embed_size,
            feature_extract=self.feature_extract,
            use_pretrained=self.use_pretrain,
            num_ft_layers=-1,
            bst_model_weights=None,
        )
        return classifier_model.to(device)

    def _build_finetune_model(self):
        """Create a finetuned model with pretrained weights"""
        classifier_model = build_models(
            model_name=self.model_name,
            num_classes=self.num_classes,
            in_channels=self.input_channels,
            embedding_size=self.embed_size,
            feature_extract=self.feature_extract,
            use_pretrained=self.use_pretrain,
            num_ft_layers=self.finetune_layer,
            bst_model_weights=self.trained_model_path,
        )
        return classifier_model.to(device)

    def _prepare_training_generators(
        self,
        train_df: pd.DataFrame,
        train_root_dir: str,
        is_kfold: bool = False,
        fold: int = -1,
    ) -> tuple:
        """
        Prepare training and validation dataloaders

        Args:
            train_df (pd.DataFrame) : Dataframe containing 2 columns `file` and `label`
            train_root_dir (str) : Base path to images if path are relative in the dataframe
            is_kfold (bool): whether to create kfold dataloaders
            fold (int) : integer corresponding to current fold number
        Returns:
            train_loader, val_loader (tuple) : A tuple of train and val dataloaders
        """

        if is_kfold:
            df_train = train_df[train_df.kfold != fold].reset_index(drop=True)
            df_val = train_df[train_df.kfold == fold].reset_index(drop=True)
            train_x, train_y = df_train["file"], df_train["label"]
            val_x, val_y = df_val["file"], df_val["label"]
        else:
            x = train_df["file"]
            y = train_df["label"]

            train_x, val_x, train_y, val_y = train_test_split(
                x,
                y,
                test_size=self.split_ratio,
                random_state=self.random_seed,
                shuffle=True,
                stratify=y,
            )
        logging.info(
            f"Training shape: {train_x.shape}, {train_y.shape}, {np.unique(train_y, return_counts=True)}"
        )
        logging.info(
            f"Validation shape: {val_x.shape}, {val_y.shape}, {np.unique(val_y, return_counts=True)}"
        )
        # augmentations
        if self.transform_type == "albu":
            trn_transforms = get_albu_train_transforms(
                self.height, self.width, self.means, self.stds
            )
            val_transforms = get_albu_val_transforms(
                self.height, self.width, self.means, self.stds
            )
        elif self.transform_type == "pt":
            trn_transforms = get_pt_train_transforms(
                self.height,
                self.width,
                self.means,
                self.stds,
                self.random_erase_prob,
                auto_augment_policy=self.auto_augment_policy,
            )
            val_transforms = get_pt_val_transforms(
                self.height, self.width, self.means, self.stds
            )
        # create dataset
        train_dataset = ImageClassifierDataset(
            img_paths=list(train_x),
            lbls=list(train_y),
            root_dir=train_root_dir,
            transform=trn_transforms,
            transform_type=self.transform_type,
        )
        val_dataset = ImageClassifierDataset(
            img_paths=list(val_x),
            lbls=list(val_y),
            root_dir=train_root_dir,
            transform=val_transforms,
            transform_type=self.transform_type,
        )
        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#mixup-and-cutmix
        collate_fn = None
        mixup_transforms = []
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(
                transforms.RandomMixup(
                    self.num_classes, p=1.0, alpha=self.mixup_alpha
                )
            )
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(
                transforms.RandomCutmix(
                    self.num_classes, p=1.0, alpha=self.cutmix_alpha
                )
            )
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            collate_fn = lambda batch: mixupcutmix(*default_collate(batch))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return train_loader, val_loader

    def prepare_df(self, ds_path: str) -> pd.DataFrame:
        """
        Prepare dataframe from raw dataset

        Args:
            ds_path (str) : Path to training csv
        Returns:
            pd.DataFrame : A dataframe containing 2 columns (`file`, `label`)
        """
        df = datasets_to_df(ds_path)
        if isinstance(df["label"].iloc[0], str):
            lbl = LabelEncoder()
            y = lbl.fit_transform(df["label"])
            self.classes = lbl.classes_
            logging.info("Label Encoding: {}".format(lbl.classes_))
            self.class_mapping = {k: v for k, v in enumerate(lbl.classes_)}
            logging.info("Label Mapping: {}".format(self.class_mapping))
            df["label"] = y
            with open(self.model_dir + "/class_mapping.json", "w") as fp:
                json.dump(self.class_mapping, fp)
        return df

    def _get_optimizers(self, train_params):
        """Choose optimizer from different options"""
        optimizer = None
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                train_params,
                lr=self.lr,
                amsgrad=True,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "radam":
            optimizer = torch.optim.RAdam(
                train_params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                train_params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                train_params,
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(
                train_params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                eps=0.0316,
                alpha=0.9,
            )
        return optimizer

    def _get_scheduler(self, optimizer, train_loader_length):
        """Choose scheduler from different options"""
        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#lr-optimizations
        # main lr schedulers
        main_lr_scheduler = None
        if self.scheduler == "onecyclelr":
            main_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr * 100,
                steps_per_epoch=train_loader_length,
                epochs=self.epochs,
            )
        elif self.scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
            )
        elif self.scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs - self.lr_warmup_epochs
            )
        elif self.scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.lr_gamma
            )
        # warmup lr schedulers
        if self.lr_warmup_epochs > 0:
            if self.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.lr_warmup_decay,
                    total_iters=self.lr_warmup_epochs,
                )
            elif self.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    start_factor=self.lr_warmup_decay,
                    total_iters=self.lr_warmup_epochs,
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[self.lr_warmup_epochs],
            )
        else:
            lr_scheduler = main_lr_scheduler
        return lr_scheduler

    @staticmethod
    def _print_stats(
        phase, epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall
    ):
        """Print training and validation stats"""

        logging.info(
            "{} {} | {} {:.4f} | {} {:.4f} | {} {:.4f} | {} {:.4f} | {} {:.4f}".format(
                colorstr("Phase:"),
                phase,
                colorstr("Loss"),
                epoch_loss,
                colorstr("Accuracy"),
                epoch_acc,
                colorstr("F1"),
                epoch_f1,
                colorstr("Precision"),
                epoch_precision,
                colorstr("Recall"),
                epoch_recall,
            )
        )

    def train_pretrain(
        self,
        cws: np.ndarray,
        train_loader_length: int,
        dataloaders_dict: dict,
        is_kfold: bool = False,
        f: int = -1,
        base_model_path: str = "",
    ) -> float:
        """
        Train a pretrained model

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length : Length of train dataloader
            dataloaders_dict : A dict containing train and val dataloaders with keys train and val
            is_kfold : whether training is using kfold dataset
            f : integer corresponding to current fold number
            base_model_path : a base path with only model name
        Returns:
            best_val_acc : Best validation accuracy
        """
        self.net = self._build_pretrain_model()

        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#weight-decay-tuning
        if self.norm_weight_decay is None:
            t_parameters = self.net.parameters()
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(
                self.net
            )
            wd_groups = [self.norm_weight_decay, self.weight_decay]
            t_parameters = [
                {"params": p, "weight_decay": w}
                for p, w in zip(param_groups, wd_groups)
                if p
            ]

        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#label-smoothing
        if self.num_classes <= 2:
            criterion = nn.BCEWithLogitsLoss(
                weight=torch.from_numpy(cws).float()
            ).to(device)
        else:
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cws).float(),
                label_smoothing=self.label_smoothing,
            ).to(device)
        optimizer = self._get_optimizers(t_parameters)
        scheduler = self._get_scheduler(optimizer, train_loader_length)

        if is_kfold:
            self.save_model_path = (
                base_model_path.split(".pth")[0] + f"_{f}_fold.pth"
            )

        trainer = Trainer(
            model=self.net,
            dataloaders=dataloaders_dict,
            num_classes=self.num_classes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=self.epochs,
            device=device,
            use_wandb=self.use_wandb,
        )

        since = time.time()
        for epoch in range(1, self.epochs + 1):
            logging.info(
                f"\n{'--' * 5} EPOCH: {epoch} | {self.epochs} {'--' * 5}\n"
            )
            (
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            ) = trainer.train_one_epoch()
            self._print_stats(
                "train",
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            )
            (
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            ) = trainer.valid_one_epoch()
            self._print_stats(
                "val",
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            )
        time_elapsed = time.time() - since
        logging.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        logging.info(colorstr("Best val Acc: {:4f}".format(trainer.best_acc)))
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        logging.info(f"Saving best pretrained model at {self.save_model_path}")
        torch.save(self.net, self.save_model_path)
        return trainer.best_acc

    def train_finetune(
        self, cws: np.ndarray, train_loader_length: int, dataloaders_dict: dict
    ) -> float:
        """
        Finetune a pretrained model
        Note: We increase the epochs to 1.5 times of the number used for pretraining and decrease the learning rate by
        10 as compared to pretraining.

        Args:
            cws (np.ndarray): A numpy array of class weights corresponding to each label
            train_loader_length (int): Length of train dataloader
            dataloaders_dict (Dict): A dict containing train and val dataloader with keys train and val
        Returns:
            best_val_acc (float): Best validation accuracy
        """
        self.lr = self.lr * 0.1
        self.epochs = int(self.epochs * 1.5)
        self.trained_model_path = self.save_model_path
        self.net = self._build_finetune_model()

        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#weight-decay-tuning
        if self.norm_weight_decay is None:
            t_parameters = self.net.parameters()
        else:
            param_groups = torchvision.ops._utils.split_normalization_params(
                self.net
            )
            wd_groups = [self.norm_weight_decay, self.weight_decay]
            t_parameters = [
                {"params": p, "weight_decay": w}
                for p, w in zip(param_groups, wd_groups)
                if p
            ]

        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#label-smoothing
        if self.num_classes <= 2:
            criterion = nn.BCEWithLogitsLoss(
                weight=torch.from_numpy(cws).float()
            ).to(device)
        else:
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cws).float(),
                label_smoothing=self.label_smoothing,
            ).to(device)
        optimizer = self._get_optimizers(t_parameters)
        scheduler = self._get_scheduler(optimizer, train_loader_length)

        trainer = Trainer(
            model=self.net,
            dataloaders=dataloaders_dict,
            num_classes=self.num_classes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=self.epochs,
            device=device,
            use_wandb=self.use_wandb,
        )

        since = time.time()
        for epoch in range(1, self.epochs + 1):
            logging.info(
                f"\n{'--' * 15} EPOCH: {epoch} | {self.epochs} {'--' * 15}\n"
            )
            (
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            ) = trainer.train_one_epoch()
            self._print_stats(
                "train",
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            )
            (
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            ) = trainer.valid_one_epoch()
            self._print_stats(
                "val",
                epoch_loss,
                epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall,
            )
        time_elapsed = time.time() - since
        logging.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        logging.info(colorstr("Best val Acc: {:4f}".format(trainer.best_acc)))
        # load best model weights
        self.net.load_state_dict(trainer.best_model_wts)
        torch.save(self.net, self.save_model_path)
        logging.info(f"Saving best finetuned model at {self.save_model_path}")
        return trainer.best_acc

    def train(self, train_df: pd.DataFrame):
        """
        Training and evaluate the model
        First perform a training of pretrained model.
        If arguments of finetuning are passed, perform a finetuning using weights of above pretrained model.
        If arguments for onnx conversion are passed, perform the conversion and calibration of the onnx model

        Args:
            train_df (pd.DataFrame) : Dataframe containing image file names and labels with columns `file` and `label`
        """
        train_y = train_df["label"]
        cws = class_weight.compute_class_weight(
            "balanced", np.unique(train_y), train_y
        )
        logging.info(f"Class weights for labels: {cws}")

        logging.info("Loading data")
        train_loader, val_loader = self._prepare_training_generators(
            train_df, self.train_root_dir
        )
        train_loader_length = len(train_loader)
        # Create training and validation dataloaders
        dataloaders_dict = {"train": train_loader, "val": val_loader}
        # train a pretrained model
        if self.feature_extract:
            logging.info(colorstr("Start training pretrained models"))
            _ = self.train_pretrain(cws, train_loader_length, dataloaders_dict)
        # finetune a pretrained model
        if self.finetune_layer != -1:
            logging.info(colorstr("Start finetuning pretrained model"))
            _ = self.train_finetune(cws, train_loader_length, dataloaders_dict)
        # convert model to onnx
        if self.convert_onnx:
            parent_model_dir = os.path.join(self.model_dir, "models")
            tracer = TraceSaver(
                self.save_model_path,
                parent_model_dir,
                (self.height, self.width, self.input_channels),
            )
            tracer.export_onnx(debug=False)
            validate_onnx = ValidateOnnx(self.model_dir)
            for tol in [1e-1, 1e-2, 1e-3]:
                logging.info(f"Performing calibration for rtol={tol}...")
                validate_onnx.calibrate(
                    (self.input_channels, self.height, self.width),
                    tol,
                    self.train_root_dir,
                )
        # perform evaluation
        self.evaluate(val_loader, cws, True, False)

    def evaluate(
        self,
        val_loader,
        cws,
        vis_prediction: bool = True,
        is_test: bool = False,
    ):
        """

        Args:
            val_loader: validation dataloader or test dataloader
            cws: numpy array of class weights corresponding to each class
            vis_prediction: if True, plot a sample of 10x2 grid with prediction and labels
            is_test: if True, log metrics with prefix `test` in wandb.ai else use prefix of `val`

        Returns:

        """
        # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/#label-smoothing
        if self.num_classes <= 2:
            criterion = nn.BCEWithLogitsLoss(
                weight=torch.from_numpy(cws).float()
            ).to(device)
        else:
            criterion = nn.CrossEntropyLoss(
                weight=torch.from_numpy(cws).float(),
                label_smoothing=self.label_smoothing,
            ).to(device)

        test_model(
            self.save_model_path,
            val_loader,
            criterion,
            self.num_classes,
            self.classes,
            device,
            self.use_wandb,
            vis_prediction,
            self.means,
            self.stds,
            is_test,
        )

    def train_k_folds(self, train_df: pd.DataFrame):
        """
        Training on stratified k-folds
        First perform a training of pretrained model.
        If arguments of finetuning are passed, perform a finetuning using weights of above pretrained model.

        Args:
            train_df (pd.DataFrame) : Dataframe containing image file names and labels
        Returns:
            best_accs (list) : A list of best val accuracy across all folds
        """
        if self.folds < 2:
            raise ValueError("Number of folds should be greater than 1")
        train_df["kfold"] = -1
        # shuffle dataset
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        skf = StratifiedKFold(n_splits=self.folds)
        # perform stratification on subclasses
        for f, (t_, v_) in enumerate(
            skf.split(X=train_df, y=train_df["label"].values)
        ):
            train_df.loc[v_, "kfold"] = f

        train_y = train_df["label"]
        cws = class_weight.compute_class_weight(
            "balanced", np.unique(train_y), train_y
        )
        logging.info(f"Class weights for labels: {cws}")

        pretrain_acc, finetune_acc, base_path = [], [], self.save_model_path
        for f in range(self.folds):
            logging.info("Loading data")
            train_loader, val_loader = self._prepare_training_generators(
                train_df, self.train_root_dir, True, f
            )
            train_loader_length = len(train_loader)
            # Create training and validation dataloaders
            dataloaders_dict = {"train": train_loader, "val": val_loader}
            logging.info(f"Training for {f} Fold")
            # train a pretrained model
            if self.feature_extract:
                logging.info(colorstr("Start training pretrained models"))
                pretrain_acc.append(
                    self.train_pretrain(
                        cws,
                        train_loader_length,
                        dataloaders_dict,
                        True,
                        f,
                        base_path,
                    )
                )
            # finetune a pretrained model
            if self.finetune_layer != -1:
                logging.info(colorstr("Start finetuning pretrained model"))
                finetune_acc.append(
                    self.train_finetune(
                        cws, train_loader_length, dataloaders_dict
                    )
                )

        if self.feature_extract:
            logging.info(
                f"Pretrained Model => Average validation accuracy across {self.folds} folds : {np.mean(pretrain_acc)}"
            )
        if self.finetune_layer != -1:
            logging.info(
                f"Finetune Model => Average validation accuracy across {self.folds} folds : {np.mean(finetune_acc)}"
            )
