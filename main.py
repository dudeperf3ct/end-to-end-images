import cv2
from train import ClassifierModel
from inference.inferPytorch import InferPytorch
from benchmark import InferenceBenchmarkRunner

if __name__ == '__main__':
    ################################
    # Model training and evaluation
    model_dir = "path/to/Param.json"
    experiment = "name-of-experiment"
    model = ClassifierModel(model_dir, experiment)
    train_df = model.prepare_df("path/to/train/csv")
    model.train(train_df)
    ################################
    # Model inference
    infer_pytorch = InferPytorch("path/to/experiment/folder")
    im = cv2.imread("path/to/sample/image")
    infer_pytorch.infer(im)
    ################################
    # Benchmark inference time
    for p in ["float32", "float16"]:
        print(f"Running inference benchmark with precision = {p}")
        benchmark = InferenceBenchmarkRunner("path/to/experiment/folder", p)
        print(benchmark.run())
    ################################
