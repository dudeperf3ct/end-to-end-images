# Almost end-to-end image classification

- [x] PyTorch training, evaluation, inference and benchmark code with SOTA practices (support for wandb.ai logging)
- [x] Onnx conversion, calibration and inference
- [x] TensorRT conversion and inference
- [x] Example notebook
- [ ] C++ Inference (Future release)



### What is this?

In this project, for a given image classification task, we can perform a large number of experiments just by changing `param.json` file.

The project supports pretraining and finetuning of `timm models`.  The training code supports scope for lot many customization for example adding more optimizers in `_get_optimizers`  or  schedulers in `_get_scheduler` functions.

It also contains an option to convert model to onnx and TensorRT. There are reference inference scripts for all different formats of the model.

----

### Getting Started

- How to run with custom dataset?
  - replace `datasets_to_df` in `utils.py` with a function that returns a dataframe with 2 columns containing image file paths named `file` and labels named `label`.
  - check if `prepare_df` in `main.py` is compatible.
- Create many different models and experiments just by replacing `model_name` in `params.json` (by creating appropriate folder for each model under `experiments` folder) or `finetune_layer` parameter or any other hyper parameter in json file.

-----

### Example

`Notebooks` folder contains a sample notebook to run `cifar10` dataset end to end.

-----

#### Modification to support regression

We can take this task further by support regression task along with classification by replacing classification based metrics, losses and some helper code with regression related code. (We can add support for a simple switch if required?)

