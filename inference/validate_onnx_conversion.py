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

    def calibrate(self, inp_shape: list, tol: float):
        """Perform calibration for pytorch and onnx model for given relative threshold

        Args:
            inp_shape: Tuple of input shape (CHW)
            tol: Relative tolerance to compare the onnx and pytorch predictions

        Returns: None

        Raises AssertionError: 
            if there's mismatch in prediction of pytorch and onnx model for given threshold

        """
        ort_session = onnxruntime.InferenceSession(self.onnx_path)
        model = torch.load(self.pytorch_path).to(device)
        model.eval()
        for _ in range(8):
            # to tensor
            rgb_tensor = torch.random.randn((1,)+(inp_shape))
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
                logging.info(f"Test Passed with relative tol={tol}")
                logging.info('-'*15)
            except AssertionError as e:
                logging.info(f"Test Failed with relative tol={tol}")
                logging.info(e)
                logging.info('-'*30)
