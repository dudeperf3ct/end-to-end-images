from tqdm import tqdm
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import wandb

from utils import plot_cm, plot_predictions, colorstr


def test_model(
    model_path: str,
    test_loader,
    criterion,
    num_classes: int,
    classes: list,
    device,
    use_wandb: bool,
    vis_prediction: bool,
    means: list,
    stds: list,
    is_test: bool,
):
    """
    Evaluate trained model

    Args:
        model_path: path to load the trained model
        test_loader: can be either a validation dataloader or test dataloader
        criterion: loss function
        num_classes: number of classes
        classes: list of classes in order with which they are encoded
        device: to place the model and inputs
        use_wandb: if True, log different metrics and images to wandb.ai
        vis_prediction: if True, show the grid of 10x2 random images along with their prediction and labels
        means: means for normalization
        stds: stds for normalization
        is_test: if True, log with test metrics else log with val metrics in wandb.ai

    Returns:

    """
    # initialize lists to monitor test loss and accuracy
    example_images, all_pred, all_lbl = [], [], []
    running_test_loss = 0.0
    class_correct = list(0.0 for i in range(num_classes))
    class_total = list(0.0 for i in range(num_classes))
    confusion_matrix = torch.zeros(num_classes, num_classes)

    # eval mode
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        stream = tqdm(
            test_loader, total=len(test_loader), position=0, leave=True
        )
        for _, (data, target) in enumerate(stream):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            if num_classes <= 2:
                onehot_labels = torch.nn.functional.one_hot(
                    target, num_classes
                )
                onehot_labels = onehot_labels.type_as(output)
                loss = criterion(output, onehot_labels)
            else:
                target = target.long()
                loss = criterion(output, target)
            stream.set_description("test_loss: {:.2f}".format(loss.item()))
            # update test loss
            running_test_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = pred.eq(target.data.view_as(pred))
            # calculate test accuracy for each object class
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            for i in range(len(data)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            if use_wandb:
                example_images.append(
                    wandb.Image(
                        data[0],
                        caption="Pred: {} Truth: {}".format(
                            classes[pred[0].item()], classes[target[0].item()]
                        ),
                    )
                )
            all_pred.append(pred.tolist())
            all_lbl.append(target.tolist())

    # calculate and print avg test loss
    test_loss = running_test_loss / len(test_loader.dataset)
    print(colorstr("Test Loss:"), "{:.4f}\n".format(test_loss))

    for i in range(num_classes):
        if class_total[i] > 0:
            print(
                "Test Accuracy of %5s: %.2f%% (%2d/%2d)"
                % (
                    str(i),
                    100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]),
                    np.sum(class_total[i]),
                )
            )
        else:
            print(
                "Test Accuracy of %5s: N/A (no training examples)"
                % (classes[i])
            )

    print(
        "\n",
        colorstr("Test Accuracy (Overall):"),
        " %.2f%% (%2d/%2d)"
        % (
            100.0 * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct),
            np.sum(class_total),
        ),
    )

    print("\n", colorstr("Confusion Matrix"), "\n", confusion_matrix.numpy())

    flat_lbl = [item for sublist in all_lbl for item in sublist]
    flat_pred = [item for sublist in all_pred for item in sublist]
    _ = plot_cm(flat_lbl, flat_pred, classes)
    plt.show()

    if use_wandb:
        if is_test:
            # log confusion matrix
            wandb.log(
                {
                    "conf_mat_test": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=flat_lbl,
                        preds=flat_pred,
                        class_names=classes,
                    )
                }
            )
            # log loss, accuracy and sample predictions
            wandb.log(
                {
                    "Examples_test": example_images,
                    "test_acccuracy": 100.0
                    * np.sum(class_correct)
                    / np.sum(class_total),
                    "test_loss_": test_loss,
                }
            )

        else:
            # log confusion matrix
            wandb.log(
                {
                    "conf_mat_val": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=flat_lbl,
                        preds=flat_pred,
                        class_names=classes,
                    )
                }
            )

            # log loss, accuracy and sample predictions
            wandb.log(
                {
                    "Examples_val": example_images,
                    "val_acccuracy": 100.0
                    * np.sum(class_correct)
                    / np.sum(class_total),
                    "val_loss_": test_loss,
                }
            )

    print(
        "\n",
        colorstr("Classification Report"),
        "\n",
        metrics.classification_report(flat_lbl, flat_pred),
    )

    if vis_prediction:
        _ = plot_predictions(model, test_loader, device, classes, means, stds)
        plt.show()
