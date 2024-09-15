import cv2
from imread_from_url import imread_from_url
from neuflowv2 import NeuFlowV2, draw_flow

# Initialize model
model_path = "models/neuflow_sintel.onnx"
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