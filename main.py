"""Sample main.py to run as python script"""
import cv2

from train import ClassifierModel
from inference.inferPytorch import InferPytorch
from benchmark import InferenceBenchmarkRunner

do_trt_inference = False

if __name__ == "__main__":
    ################################
    # Model training and evaluation
    model_dir = "path/to/Param.json"
    experiment = "name-of-experiment"
    model = ClassifierModel(model_dir, experiment)
    train_df = model.prepare_df("path/to/train/csv")
    # perform k-fold training
    if model.folds > 1:
        model.train_k_folds(train_df)
    # perform simple training
    else:
        model.train(train_df)
    ################################
    # Model inference
    infer_pytorch = InferPytorch(model_dir)
    im = cv2.imread("path/to/sample/image")
    infer_pytorch.infer(im)
    ################################
    # Benchmark inference time
    for p in ["float32", "float16"]:
        print(f"Running inference benchmark with precision = {p}")
        benchmark = InferenceBenchmarkRunner(model_dir, p)
        print(benchmark.run())
    ################################

    if do_trt_inference:
        # run only this part in Docker.trt
        from inference.inferTensorRT import InferTensorRT

        infer_trt = InferTensorRT(model_dir, "fp16")
        infer_trt.infer_img("/path/to/sample/image", 3)
