import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import time
import os
import json
import cv2

from inference.trtCalibINT8 import EntropyCalibrator


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self) -> str:
        return str(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class HostDeviceMem(object):
    """Simple helper data class for storing memory references"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __repr__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

class InferTensorRT:
    def __init__(self, model_dir: str, engine_file_path: str = None, batch_size: int = 1, mode: str = "fp16", calib_dir: str = None):
        """
        Build a tensorrt engine

        Args:
            model_dir : Path to model directory containing models folder
            engine_file_path : Optional, if engine already built it will be loaded
            batch_size : Batch size for inference
            mode : Mode for building a tensorrt enine
            calib_dir : Required if `int8` calibration needs to be performed 
        """
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.cuda_ctx = cuda.Device(0).make_context() # Use GPU:0
        self.runtime = trt.Runtime(self.TRT_LOGGER)

        self.model_path = os.path.join(model_dir, "models/")
        json_path = os.path.join(model_dir, "params.json")
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = Params(json_path)
        onnx_file_path = os.path.join(self.model_path, "model.onnx")

        # inputs and outputs shapes
        data_h, data_w, data_c = params.height, params.width, params.input_channels
        self.data_h, self.data_w, self.data_d = data_h, data_w, data_c
        self.num_classes = params.num_classes
        means, stds = params.means, params.stds
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        # load class label mapping
        with open(model_dir + "/class_mapping.json") as fp:
            self.class_mapping = json.load(fp)

        self.engine_file_path = self.get_engine_file_path(onnx_file_path, engine_file_path, batch_size, mode)
        self.calib_dir = calib_dir
        
        if os.path.exists(self.engine_file_path):
            self.engine = self.load_engine(self.engine_file_path)
        else:
            # buil TensorRT engine from ONNX model
            self.engine = self.build_engine(onnx_file_path, self.engine_file_path, max_batch_size=batch_size, mode=mode)
        
        try:
            self.context = self.create_context(self.engine)
            self.inputs, self.outputs, self.bindings, self.output_shapes, self.stream = self.allocate_buffers(self.engine)
        except Exception as e:
            self.cuda_ctx.pop()
            del self.cuda_ctx
            raise RuntimeError("Fail to allocate CUDA resources") from e
    
    def __del__(self):
        """Free CUDA memories"""
        del self.stream
        del self.outputs
        del self.inputs
        self.cuda_ctx.pop()
        del self.cuda_ctx
    
    def create_context(self, engine=None):
        engine = engine or self.engine
        return engine.create_execution_context()
    
    def get_engine_file_path(self, onnx_file_path, engine_file_path, batch_size, mode):
        if not engine_file_path:
            engine_file_path, _ = os.path.splitext(onnx_file_path)
            engine_file_path = "{}_{}_b{}.trt".format(engine_file_path, mode, batch_size)
        return engine_file_path
    
    def allocate_buffers(self, engine):
        """Allocates all host/device in/out buffers required for an engine."""

        inputs   = []
        outputs  = []
        bindings = []
        output_shapes = []
        stream   = cuda.Stream()    
        for binding in engine:
            size  = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers.
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes) 
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                output_shapes.append(engine.get_binding_shape(binding))
                outputs.append(HostDeviceMem(host_mem, device_mem))
            
        return inputs, outputs, bindings, output_shapes, stream
    
    def build_engine(self, onnx_file_path, engine_file_path, max_batch_size=1, max_workspace_size=30, mode="fp16"):
        st = time.time()
        
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, self.TRT_LOGGER) as parser:
            
            # parse ONNX file
            print("Loading ONNX file: '{}'".format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                parser.parse(model.read())
                
                input_shape = network.get_layer(0).get_input(0).shape
                # input_layer_name = network.get_layer(0).name
                input_layer_name = network.get_layer(0).get_input(0).name
                
                net_h, net_w = input_shape[2], input_shape[3]
                last_layer = network.get_layer(network.num_layers - 1)
                
                # Check if last layer recognizes it's output
                if not last_layer.get_output(0):
                    # If not, then mark the output using TensorRT API
                    network.mark_output(last_layer.get_output(0))
            print('Completed parsing of ONNX file')
            
            builder.max_batch_size = max_batch_size
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            
            profile = builder.create_optimization_profile()
            
            profile.set_shape(
                input_layer_name,                   # input tensor name
                (max_batch_size, 3, net_h, net_w),  # min shape
                (max_batch_size, 3, net_h, net_w),  # opt shape
                (max_batch_size, 3, net_h, net_w))  # max shape
            
            config.add_optimization_profile(profile)
            
            # use FP16 mode
            if mode == "fp16" and builder.platform_has_fast_fp16:
                print("converting to fp16")
                builder.fp16_mode = True
                
            elif mode == "int8" and builder.platform_has_fast_int8:
                print("converting to int8")
                
                config.set_flag(trt.BuilderFlag.INT8)
                calib_images = os.listdir(self.calib_dir)
                calib_dataset = [os.path.join(self.calib_dir, x) for x in calib_images]
                config.int8_calibrator = EntropyCalibrator(model_path=self.model_path,
                                        calibration_files=calib_dataset,
                                        batch_size=1,
                                        h=self.data_h,
                                        w=self.data_w,
                                        means=self.means,
                                        stds=self.stds)
                config.set_calibration_profile(profile)
            
            
            # generate TensorRT engine optimized for the target platform
            print('Building an Engine...')
            engine = builder.build_engine(network, config)
            print("Completed creating Engine")
            print("Elapsed: {:.3f} sec".format(time.time() - st))

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
    
    def load_engine(self, engine_file_path, runtime=None):
        runtime = runtime or self.runtime
        print("Loading Engine: '{}'".format(engine_file_path))
        with open(engine_file_path, 'rb') as f:
            engine_data = f.read()
        
        engine = runtime.deserialize_cuda_engine(engine_data)
        print("Completed loading Engine...")
        return engine
    
    def run(self, batch):
        self.inputs[0].host = np.array(batch, dtype=np.float32, order='C')
        trt_outputs = self.do_inference(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream
            )
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]        
        return trt_outputs
    
    def do_inference(self, context, bindings, inputs, outputs, stream):
        """do_inference (for TensorRT 7.0+)

        This function is generalized for multiple inputs/outputs for full
        dimension networks.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def preprocess_input(self, rgb_img):
        # permute and normalize
        rgb_tensor = (rgb_img.astype(np.float32) / 255.0 - self.means) / self.stds
        rgb_tensor = np.transpose(rgb_tensor, (2, 0, 1))
        # add batch dimension
        rgb_tensor = rgb_tensor[np.newaxis, ...]
        return rgb_tensor
    
    def infer_img(self, img_path: str, topk:int = 3, verbose:bool = True):
        """
        Perform TensorRT inference on given image
        Args:
            img_path : Path to sample image
            topk : Number of topk classes to return
            verbose : Print intermediate outputs
        
        Returns:
            Topk indices
        """
        bgr_img = cv2.imread(img_path)
        if verbose:
            print(f"Resized image to {(self.data_w, self.data_h, self.data_d)} from {(bgr_img.shape)}")
        # resize
        bgr_img = cv2.resize(bgr_img, (self.data_w, self.data_h), interpolation=cv2.INTER_LINEAR)
        # get make rgb
        if verbose:
            print("Converting bgr to rgb")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        batch = self.preprocess_input(rgb_img)
        out = np.array(self.run(batch)[0]).flatten()
        topk_ind =  np.argsort(out)[::-1][:topk]
        if verbose:
            print(f"Getting {topk} classes")
            for ind in topk_ind:
                print("{} {} ({})".format(out[ind], ind, self.class_mapping[str(ind)]))
        return topk_ind