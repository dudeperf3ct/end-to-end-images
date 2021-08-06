import copy
import logging
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import wandb
import torch
try:
    from torch.cuda.amp import autocast, GradScaler
except AttributeError:
    raise AttributeError("AMP training is used as default")
from torch.cuda.amp import autocast, GradScaler


# cool tricks: https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/
class Trainer:
    """Trainer with training and validation loops with amp precision"""
    def __init__(self, model, dataloaders: dict, num_classes: int, criterion,
                 optimizer, scheduler, num_epochs: int, device, use_wandb: bool):
        """
            Args:
                model : PyTorch model
                dataloaders (dict) : Dict containing train and val dataloaders
                num_classes (int) : Number of classes to one hot targets if num_classes <= 2
                criterion : pytorch loss function
                optimizer : pytorch optimizer function
                scheduler : pytorch scheduler function
                num_epochs (int) : Number of epochs to train the model
                device : torch.device indicating whether device is cpu or gpu
                use_wandb (bool) : Log results to wandb.ai
        """
        self.model = model
        self.train_data = dataloaders['train']
        self.valid_data = dataloaders['val']
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.use_wandb = use_wandb
        self.best_acc = 0.0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        # create a GradScaler once at the beginning of training.
        self.scaler = GradScaler()

    def train_one_epoch(self):
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0.0
        f1s, recalls, precisions = [], [], []

        stream = tqdm(self.train_data, position=0, leave=True)
        # Iterate over batch of data.
        for _, (inputs, labels) in enumerate(stream):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # zero the parameter gradients
            self.optimizer.zero_grad(set_to_none=True)
            # Runs the forward pass with autocasting.
            with autocast():
                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    # backward + optimize only if in training phase
                    outputs = self.model(inputs)
                    if self.num_classes <= 2:
                        onehot_labels = torch.nn.functional.one_hot(labels, self.num_classes)
                        onehot_labels = onehot_labels.type_as(outputs)
                        loss = self.criterion(outputs, onehot_labels)
                    else:
                        labels = labels.long()
                        loss = self.criterion(outputs, labels)
                    stream.set_description('train_loss: {:.2f}'.format(loss.item()))
                    # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                    self.scaler.scale(loss).backward()
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            # statistics
            _, preds = torch.max(outputs, 1)
            f1s.append(
                metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            precisions.append(
                metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            recalls.append(
                metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        self.scheduler.step()
        epoch_loss = running_loss / (len(self.train_data.dataset))
        epoch_acc = (running_corrects.double() / (len(self.train_data.dataset))).item()
        epoch_f1 = np.mean(f1s)
        epoch_precision = np.mean(precisions)
        epoch_recall = np.mean(recalls)
        # log to wandb
        if self.use_wandb:
            wandb.log(
                {
                    f"train_epoch_loss": epoch_loss,
                    f"train_epoch_accuracy": epoch_acc,
                    f"train_epoch_f1": epoch_f1,
                    f"train_epoch_precision": epoch_precision,
                    f"train_epoch_recall": epoch_recall
                }
            )
        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall

    def valid_one_epoch(self):
        self.model.eval()  # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0.0
        f1s, recalls, precisions = [], [], []

        stream = tqdm(self.valid_data, position=0, leave=True)
        # Iterate over batch of data.
        for _, (inputs, labels) in enumerate(stream):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                if self.num_classes <= 2:
                    onehot_labels = torch.nn.functional.one_hot(labels, self.num_classes)
                    onehot_labels = onehot_labels.type_as(outputs)
                    loss = self.criterion(outputs, onehot_labels)
                else:
                    labels = labels.long()
                    loss = self.criterion(outputs, labels)
                stream.set_description('val_loss: {:.2f}'.format(loss.item()))
            # statistics
            _, preds = torch.max(outputs, 1)
            f1s.append(
                metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            precisions.append(
                metrics.precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            recalls.append(
                metrics.recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
            )
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / (len(self.valid_data.dataset))
        epoch_acc = (running_corrects.double() / (len(self.valid_data.dataset))).item()
        epoch_f1 = np.mean(f1s)
        epoch_precision = np.mean(precisions)
        epoch_recall = np.mean(recalls)
        # log to wandb
        if self.use_wandb:
            wandb.log(
                {
                    f"val_epoch_loss": epoch_loss,
                    f"val_epoch_accuracy": epoch_acc,
                    f"val_epoch_f1": epoch_f1,
                    f"val_epoch_precision": epoch_precision,
                    f"val_epoch_recall": epoch_recall
                }
            )
        # deep copy the model
        if epoch_acc > self.best_acc:
            logging.info("Val acc improved from {:.4f} to {:.4f}.".format(self.best_acc, epoch_acc))
            self.best_acc = epoch_acc
            self.best_model_wts = copy.deepcopy(self.model.state_dict())
        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall
