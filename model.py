import copy
import torch
import torch.nn as nn

import timm


def set_parameter_requires_grad(
    model, feature_extracting: bool, num_ft_layers: int
):
    """
    Freeze the weights of the model is feature_extracting=True
    Fine tune layers >= num_ft_layers
    Batch Normalization: https://keras.io/guides/transfer_learning/

    Args:
        model: PyTorch model
        feature_extracting (bool): A bool to set all parameters to be trainable or not
        num_ft_layers (int): Number of layers to freeze and unfreezing the rest
    """
    if feature_extracting:
        if num_ft_layers != -1:
            for i, module in enumerate(model.modules()):
                if i >= num_ft_layers:
                    if not isinstance(module, nn.BatchNorm2d):
                        module.requires_grad_(True)
                else:
                    module.requires_grad_(False)
        else:
            for param in model.parameters():
                param.requires_grad = False
    # not recommended to set feature_extracting=True when use_pretrained=True
    else:
        for param in model.parameters():
            param.requires_grad = True


def _create_classifier(num_ftrs: int, embedding_size: int, num_classes: int):
    """Add a classifier head with 2 FC layers

    Args:
        num_ftrs (int): Number of features from timm models
        embedding_size (int): Number of features in penultimate layer
        num_classes (int): Number of classes
    """
    head = nn.Sequential(
        nn.Linear(num_ftrs, embedding_size),
        nn.Linear(embedding_size, num_classes),
    )
    return head


def build_models(
    model_name: str,
    num_classes: int,
    in_channels: int,
    embedding_size: int,
    feature_extract: bool = True,
    use_pretrained: bool = True,
    num_ft_layers: int = -1,
    bst_model_weights=None,
):
    """
    Build various architectures to either train from scratch, finetune or as feature extractor.

    Args:
        model_name (str) : Name of model from `timm.list_models(pretrained=use_pretrained)`
        num_classes (int) : Number of output classes added as final layer
        in_channels (int) : Number of input channels
        embedding_size (int): Size of intermediate features
        feature_extract (bool): Flag for feature extracting.
                               False = finetune the whole model,
                               True = only update the new added layers params
        use_pretrained (bool): Pretraining parameter to pass to the model or if base_model_path is given use that to
                                initialize the model weights
        num_ft_layers (int) : Number of layers to finetune
                             Default = -1 (do not finetune any layers)
        bst_model_weights : Best weights obtained after training pretrained model
                            which will be used for further finetuning.

    Returns:
        model : A pytorch model
    """
    supported_models = timm.list_models(pretrained=use_pretrained)
    model = None
    if model_name in supported_models:
        model = timm.create_model(
            model_name, pretrained=use_pretrained, in_chans=in_channels
        )
        set_parameter_requires_grad(model, feature_extract, num_ft_layers)
        # check if last layer in timm models is either classifier or fc
        try:
            num_ftrs = model.classifier.in_features
            model.classifier = _create_classifier(
                num_ftrs, embedding_size, num_classes
            )
        except AttributeError:
            num_ftrs = model.fc.in_features
            model.fc = _create_classifier(
                num_ftrs, embedding_size, num_classes
            )
    else:
        print("Invalid model name, exiting...")
        exit()
    # load best model dict for further finetuning
    if bst_model_weights is not None:
        pretrain_model = torch.load(bst_model_weights)
        best_model_wts = copy.deepcopy(pretrain_model.state_dict())
        if feature_extract and num_ft_layers != -1:
            model.load_state_dict(best_model_wts)
    return model
