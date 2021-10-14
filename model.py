import torch
import numpy as np
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ImagePrediction:

    def __init__(self) -> None:
        self.model = torch.load('resnet18_finetune.pth', map_location=device).eval()
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def preprocess(self, img):
        # normalize RGB image
        rgb_arr = (np.array(img).astype(np.float32) / 255.0 - self.means) / self.stds
        # HWC -> CHW
        rgb_tensor = torch.from_numpy(rgb_arr)
        rgb_tensor = np.transpose(rgb_tensor, (2, 0, 1))
        # add batch dimension
        rgb_tensor = rgb_tensor[np.newaxis, ...]
        return rgb_tensor

    def predict(self, img: Image.Image):
        img_tensor = self.preprocess(img)
        with torch.no_grad():
            output = self.model(img_tensor.to(device))
        return {"prediction": int(output)}
