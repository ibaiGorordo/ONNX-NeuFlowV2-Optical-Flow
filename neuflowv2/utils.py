import os
import tqdm
import requests
import cv2
import numpy as np
from .flow_plot import flow_to_image

available_models = ["neuflow_mixed", "neuflow_sintel", "neuflow_things"]

def download_model(url: str, path: str):
    print(f"Downloading model from {url} to {path}")
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm.tqdm(r.iter_content(chunk_size=1024 * 1024), total=total_length // (1024 * 1024),
                               bar_format='{l_bar}{bar:10}'):
            if chunk:
                f.write(chunk)
                f.flush()


def check_model(model_path: str):
    if os.path.exists(model_path):
        return

    model_name = os.path.basename(model_path).split('.')[0]
    if model_name not in available_models:
        raise ValueError(f"Invalid model name: {model_name}")
    url = f"https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases/download/0.1.0/{model_name}.onnx"
    download_model(url, model_path)

def draw_flow(flow, image, boxes=None):
    flow_img = flow_to_image(flow, 35)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

    combined = cv2.addWeighted(image, 0.5, flow_img, 0.6, 0)
    if boxes is not None:
        white_background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        new_image = cv2.addWeighted(image, 0.7, white_background, 0.4, 0)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            new_image[y1:y2, x1:x2] = combined[y1:y2, x1:x2]

        combined = new_image

    return combined