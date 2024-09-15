# ONNX NeuFlowV2 Optical Flow Estimation
 Python scripts performing optical flow estimation using the NeuFlowV2 model in ONNX.
 
![!ONNX NeuFlowV2 Optical Flow Estimation]()

## Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

## Installation
```bash
git clone https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow.git
cd ONNX-NeuFlowV2-Optical-Flow
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

## ONNX model
- If the model file is not found in the models directory, it will be downloaded automatically from the [Release Assets](https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases/tag/0.1.0).
- **Available models**: neuflow_mixed.onnx, neuflow_sintel.onnx, neuflow_things.onnx

## Original NeuflowV2 model
The original NeuflowV2 model can be found in this repository: https://github.com/neufieldrobotics/NeuFlow_v2
- The License of the models is Apache-2.0 license: https://github.com/neufieldrobotics/NeuFlow_v2/blob/master/LICENSE

## Examples

 * **Image inference**:
 ```
 python image_optical_flow.py
 ```

 * **Webcam inference**:
 ```
 python webcam_optical_flow.py
 ```

 * **Video inference**:
 ```
 python video_optical_flow.py
 ```
![!neuflowv2_optical_flow_video]()

 * **Optical Flow of Detected Objects**: https://youtu.be/S0RnlEHGNrc
Install first YOLOv10 detection library: `pip install yolov10-onnx`
 ```
 python video_detection_optical_flow.py
 ```
![!neuflowv2_optical_flow_video]()

## References:
* NeuflowV2 model: https://github.com/neufieldrobotics/NeuFlow_v2