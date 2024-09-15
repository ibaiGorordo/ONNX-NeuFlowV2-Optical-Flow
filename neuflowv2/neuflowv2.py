import time
import cv2
import numpy as np
import onnxruntime

from .utils import check_model

class NeuFlowV2:

    def __init__(self, path: str):
        check_model(path)

        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, img_prev: np.ndarray, img_now: np.ndarray) -> np.ndarray:
        return self.estimate_flow(img_prev, img_now)

    def estimate_flow(self, img_prev: np.ndarray, img_now: np.ndarray) -> np.ndarray:
        input_tensors = self.prepare_inputs(img_prev, img_now)

        # Perform inference on the image
        outputs = self.inference(input_tensors)

        return self.process_output(outputs[0])

    def prepare_inputs(self, img_prev: np.ndarray, img_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.img_height, self.img_width = img_now.shape[:2]

        input_prev = self.prepare_input(img_prev)
        input_now = self.prepare_input(img_now)

        return input_prev, input_now

    def prepare_input(self, img: np.ndarray) -> np.ndarray:
        # Resize input image
        input_img = cv2.resize(img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensors: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensors[0],
                                                       self.input_names[1]: input_tensors[1]})

        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, output) -> np.ndarray:
        flow = output.squeeze().transpose(1, 2, 0)

        return cv2.resize(flow, (self.img_width, self.img_height))

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from imread_from_url import imread_from_url
    from utils import draw_flow

    # Initialize model
    model_path = "../models/neuflow_sintel.onnx"
    estimator = NeuFlowV2(model_path)

    # Load images
    img1 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0016.png?raw=true")
    img2 = imread_from_url("https://github.com/princeton-vl/RAFT/blob/master/demo-frames/frame_0025.png?raw=true")

    # Estimate optical flow
    flow = estimator(img1, img2)

    # Draw Flow
    flow_img = draw_flow(flow, img1)

    cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
    cv2.imshow("Optical Flow", flow_img)
    cv2.waitKey(0)