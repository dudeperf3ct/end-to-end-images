# Adapted from: https://github.com/PRBonn/bonnetal/blob/master/train/tasks/segmentation/modules/traceSaver.py
import os
import errno
import onnx
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TraceSaver:
    def __init__(self, model_path: str, new_path: str, force_img_prop: tuple = (None, None, None)):
        """

        Args:
            model_path: Path where pytorch model is saved
            new_path: Path to folder where onnx model will be saved as `model.onnx`
            force_img_prop:  tuple containing (height, width, channels)
        """
        self.path = model_path
        self.new_path = new_path
        if force_img_prop[0] is not None and force_img_prop[1] is not None and force_img_prop[2] is not None:
            self.height = force_img_prop[0]
            self.width = force_img_prop[1]
            self.channels = force_img_prop[2]
            print("WARNING: FORCING IMAGE PROPERTIES TO")
            print("(Height, Width, Channels: ({}, {}, {}))".format(self.height, self.width, self.channels))
        if os.path.exists(self.path):
            self.model = torch.load(self.path)
        else:
            print(f"No trained model found at {self.path}")
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

        # CUDA speedup
        torch.backends.cudnn.benchmark = True
        self.model = self.model.to(device)

        # eval mode
        self.model.eval()
        print("Creating dummy input to profile")
        self.dummy_input = torch.randn(1, self.channels, self.height, self.width).to(device)

    def export_onnx(self, debug: bool = True):
        """
        Convert to onnx model with given height and width

        Args:
            debug: print the onnx model graph

        Returns:
        Save onnx model to folder specified in `new_path` as `model.onnx`
        """
        # create profile
        onnx_path = os.path.join(self.new_path, "model.onnx")
        with torch.no_grad():
            print("Profiling model")
            print("Saving model at ", onnx_path)
            torch.onnx.export(self.model, self.dummy_input, onnx_path)
        # check that it worked
        print("Checking that it all worked out")
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        # Print a human readable representation of the graph
        if debug:
            print(onnx.helper.printable_graph(model_onnx.graph))

    def export_dynamic_onnx(self, debug: bool = True):
        """
        Dynamic model onnx used to save a onnx model with different (height, weight) than
        (height, width) used for training model

        Args:
            debug: print the onnx model graph

        Returns:
            save dynamic onnx model to folder specified in `new_path` as `model_dynamic.onnx`
        """
        # convert to a dynamic ONNX traced model
        # NOTE: Dynamic width/height may not achieve the expected performance improvement
        # with some backend such as TensorRT though.
        input_names = ['input']
        output_names = ['output']
        dynamic_axes = {'input': {0: 'batch_size', 2: 'width', 3: 'height'},
                        'output': {0: 'batch_size', 2: 'width', 3: 'height'}}
        # create profile
        onnx_path = os.path.join(self.new_path, "model_dynamic.onnx")
        with torch.no_grad():
            print("Profiling model")
            print("Saving model at ", onnx_path)
            torch.onnx.export(self.model, self.dummy_input, onnx_path, input_names, output_names, dynamic_axes)
        # check that it worked
        print("Checking that it all worked out")
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)
        # Print a human readable representation of the graph
        if debug:
            print(onnx.helper.printable_graph(model_onnx.graph))
