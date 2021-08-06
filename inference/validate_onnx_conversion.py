import os
import cv2
import logging
import numpy as np
import torch
import onnxruntime

from utils import Params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValidateOnnx:
    """Validate the prediction of onnx model with the pytorch model"""

    def __init__(self, model_dir: str):
        """
        Load the model with path corresponding to `save_model_path` in `param.json` file.

        Args:
            model_dir:  Path to folder containing `params.json` file and `model.onnx` in `models` directory.
        """
        json_path = os.path.join(model_dir, "params.json")
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)
        self.model_path = os.path.join(model_dir, "models/")
        self.onnx_path = os.path.join(self.model_path, 'model.onnx')

        self.pytorch_path = params.save_model_path
        self.data_h, self.data_w, self.data_d = params.height, params.width, params.input_channels
        means, stds = params.means, params.stds
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    @staticmethod
    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def calibrate(self, img_paths: list, tol: float, root_dir: str = None):
        """Perform calibration for pytorch and onnx model for given relative threshold

        Args:
            img_paths: List of some sample image paths to calibrate the prediction
            tol: Relative tolerance to compare the onnx and pytorch predictions
            root_dir: Root directory to load images if the paths are not absolute

        Returns: None
        Prints the assertion output if there's mismatch in prediction of pytorch and onnx model for given threshold

        """
        ort_session = onnxruntime.InferenceSession(self.onnx_path)
        model = torch.load(self.pytorch_path).to(device)
        model.eval()
        for img_path in img_paths:
            if root_dir is not None:
                bgr_img = cv2.imread(os.path.join(root_dir, img_path))
            else:
                bgr_img = cv2.imread(img_path)
            bgr_img = cv2.resize(bgr_img, (self.data_w, self.data_h), interpolation=cv2.INTER_LINEAR)
            # check if network is RGB or mono
            if self.data_d == 3:
                # get make rgb
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            elif self.data_d == 1:
                # get grayscale
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

            # pytorch results
            torch_out = model(rgb_tensor.to(device))

            # compute ONNX Runtime output prediction
            ort_inputs = {ort_session.get_inputs()[0].name: self._to_numpy(rgb_tensor)}
            ort_outs = ort_session.run(None, ort_inputs)
            try:
                # compare ONNX Runtime and PyTorch results
                np.testing.assert_allclose(self._to_numpy(torch_out), ort_outs[0], rtol=tol, atol=1e-05)
                logging.info(f"Passed {img_path} with relative tol={tol}")
                logging.info('-'*15)
            except AssertionError as e:
                logging.info(f"Failed {img_path} with relative tol={tol}")
                logging.info(e)
                logging.info('-'*30)
