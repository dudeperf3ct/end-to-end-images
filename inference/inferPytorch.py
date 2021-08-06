import time
import cv2
import json
import os
import numpy as np
import torch

from utils import Params, colorstr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferPytorch:
    """Pytorch inference"""

    def __init__(self, model_dir: str):
        """
        Load the model with path corresponding to `save_model_path` in `param.json` file.

        Args:
            model_dir:  Path to folder containing `params.json` file
        """
        json_path = os.path.join(model_dir, "params.json")
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)

        # some useful data
        self.data_h, self.data_w, self.data_d = params.height, params.width, params.input_channels
        self.num_classes = params.num_classes
        means, stds = params.means, params.stds
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

        # load the model
        try:
            self.pytorch_path = params.save_model_path
            self.model = torch.load(self.pytorch_path)
            print("Successfully Pytorch model from ", self.pytorch_path)
        except Exception as e:
            print("Couldn't load Pytorch network. Error: ", e)
            quit()

        self.model.to(device)
        # load class label mapping
        with open(model_dir + "/class_mapping.json") as fp:
            self.class_mapping = json.load(fp)

    def infer(self, bgr_img, verbose: bool = True) -> str:
        """Perform inference on single image

        Args:
            bgr_img: a cv2 image read using `cv2.imread`
            verbose: if True print all details such as time taken for inference, probabilities of all classes and
                    predicted class

        Returns: Predicted class

        """
        # get sizes
        original_h, original_w, original_d = bgr_img.shape

        # resize
        bgr_img = cv2.resize(bgr_img, (self.data_w, self.data_h), interpolation=cv2.INTER_LINEAR)

        # check if network is RGB or mono
        if self.data_d == 3:
            # get make rgb
            if verbose:
                print("Converting bgr to rgb")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        elif self.data_d == 1:
            # get grayscale
            if verbose:
                print("Converting to grayscale")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        else:
            raise NotImplementedError("Network has to have 1 or 3 channels. Anything else must be implemented.")
        # to tensor
        rgb_tensor = torch.from_numpy(rgb_img)
        # permute and normalize
        rgb_tensor = (rgb_tensor.float() / 255.0 - self.means) / self.stds
        rgb_tensor = rgb_tensor.permute(2, 0, 1)
        # add batch dimension
        rgb_tensor = rgb_tensor.unsqueeze(0)
        # infer
        start = time.time()
        logits = self.model(rgb_tensor.to(device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        argmax = logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        probs = torch.nn.functional.softmax(logits[0], dim=0).detach().cpu().numpy()
        time_to_infer = time.time() - start
        # time
        if verbose:
            print("Time to infer: {:.3f}s".format(time_to_infer))
            print("{:15s} | {:15s}".format(colorstr("Predictions"), colorstr("Probabilities")))
            print("-" * 30)
            for i in range(len(probs)):
                print("{:15s} | {:.2f}%".format(self.class_mapping[str(i)], probs[i] * 100))
            print("Top-1 Prediction: {} | Probability: {:.2f}%".format(colorstr(self.class_mapping[str(argmax)]),
                                                                       probs[argmax] * 100))
        return self.class_mapping[str(argmax)]
