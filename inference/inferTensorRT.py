import time
import cv2
import json
import os
import numpy as np

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from trtCalibINT8 import EntropyCalibrator
from utils import Params

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class InferTensorRT:
    def __init__(self, model_dir: str,  mode: str, calib_dir: str = None):
        """
        Convert to TensorRT model and use it for inference

        Args:
            model_dir: Path to parent folder containing onnx model stored inside `models` folder with name `model.onnx`
            mode: Mode for tensorrt conversion [fp16, int8, both]
            calib_dir: Path to folder containing sample images that will be used for int8 calibration. [Default = None]
        """
        self.model_path = os.path.join(model_dir, "models/")
        json_path = os.path.join(model_dir, "params.json")
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)

        data_h, data_w, data_c = params.height, params.width, params.input_channels
        self.num_classes = params.num_classes
        means, stds = params.means, params.stds
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

        # load class label mapping
        with open(model_dir + "/class_mapping.json") as fp:
            self.class_mapping = json.load(fp)

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.engine_serialized_path = os.path.join(self.model_path, f"model_mode_{mode}.trt")

        calib_images = os.listdir(calib_dir)
        calib_dataset = [os.path.join(calib_dir, x) for x in calib_images]

        # Determine dimensions and create CUDA memory buffers
        # to hold host inputs/outputs.
        self.data_h, self.data_w, self.data_d = data_h, data_w, data_c
        d_input_size = self.data_h * self.data_w * self.data_d * 4
        d_output_size = self.num_classes * 4
        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(d_input_size)
        self.d_output = cuda.mem_alloc(d_output_size)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

        # make a builder
        builder = trt.Builder(TRT_LOGGER)

        # support int 8 calibration?
        print("Platform has int8 mode: ", builder.platform_has_fast_int8)
        # the calibrator
        print("Trying to calibrate int8 from list of images")
        self.engine = None  # force rebuild of engine
        calib = None
        if mode == 'int8' or mode == 'both':
            calib = EntropyCalibrator(model_path=self.model_path,
                                      calibration_files=calib_dataset,
                                      batch_size=1,
                                      h=self.data_h,
                                      w=self.data_w,
                                      means=self.means,
                                      stds=self.stds)

        # architecture definition from onnx if no engine is there
        # get weights?
        if self.engine is None:
            try:
                # basic stuff for onnx parser
                model_path = os.path.join(self.model_path, "model.onnx")
                network = builder.create_network(network_flags)
                onnxparser = trt.OnnxParser(network, TRT_LOGGER)
                model = open(model_path, 'rb')
                onnxparser.parse(model.read())
                print("Successfully ONNX weights from ", model_path)
            except Exception as e:
                print("Couldn't load ONNX network. Error: ", e)
                quit()

            print("Wait while tensorRT profiles the network and build engine")
            # trt parameters
            try:
                builder.max_batch_size = 1
                builder.max_workspace_size = 8000000000
                builder.int8_mode = False
                builder.fp16_mode = False
                if mode == "int8":
                    builder.int8_calibrator = calib
                    builder.int8_mode = True
                if mode == "both":
                    builder.int8_calibrator = calib
                    builder.fp16_mode = True
                    builder.int8_mode = True
                if mode == "fp16":
                    builder.fp16_mode = True
                print("-" * 80)
                print(f"Performing {mode} conversion")
                print("-" * 80)
                print("Calling build_cuda_engine")
                self.engine = builder.build_cuda_engine(network)
                assert (self.engine is not None)
            except Exception as e:
                print("Failed creating self.engine for TensorRT. Error: ", e)
                quit()
            print("Done generating tensorRT self.engine.")

            # serialize for later
            print("Serializing tensorRT self.engine for later (for example in the C++ interface)")
            try:
                self.serialized_engine = self.engine.serialize()
                with open(self.engine_serialized_path, "wb") as f:
                    f.write(self.serialized_engine)
            except Exception as e:
                print("Couldn't serialize self.engine. Not critical, so I continue. Error: ", e)
        else:
            print("Successfully opened self.engine from inference directory.")
            print("WARNING: IF YOU WANT TO PROFILE FOR THIS COMPUTER DELETE model.trt FROM THAT DIRECTORY")

        # create execution context
        self.context = self.engine.create_execution_context()

    def infer(self, bgr_img, topk: int = 1, verbose: bool = True):
        """

        Args:
            bgr_img:
            topk:
            verbose:

        Returns:

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

        # permute and normalize
        rgb_tensor = (rgb_img.astype(np.float32) / 255.0 - self.means) / self.stds
        rgb_tensor = np.transpose(rgb_tensor, (2, 0, 1))

        # add batch dimension
        rgb_tensor = rgb_tensor[np.newaxis, ...]

        # placeholders
        h_input = np.ascontiguousarray(rgb_tensor, dtype=np.float32)
        h_output = np.empty(self.num_classes, dtype=np.int32, order='C')

        # infer
        start = time.time()

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.d_input, h_input, self.stream)
        # Run inference.
        self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()

        logits = h_output.reshape((self.num_classes))
        max_k = np.argpartition(logits, -topk)[-topk:]
        time_to_infer = time.time() - start

        # print result, and put in output lists
        classes = []
        classes_str = []
        if verbose:
            print("Time to infer: {:.3f}s".format(time_to_infer))
        for i, idx in enumerate(max_k):
            class_string = self.class_mapping[str(idx)]
            classes.append(idx)
            classes_str.append(class_string)
            if verbose:
                print("[{}]: {}, {}".format(i, idx, class_string))
        return classes, classes_str
