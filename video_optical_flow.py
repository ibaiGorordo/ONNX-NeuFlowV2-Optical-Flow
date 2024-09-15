from datetime import timedelta
import cv2
from cap_from_youtube import cap_from_youtube

from neuflowv2 import NeuFlowV2, draw_flow

# Initialize video
videoUrl = 'https://youtu.be/gbmGzQe7F_w?si=06qmqv5XbvlvwgbC'
start_time = timedelta(seconds=5)
cap = cap_from_youtube(videoUrl, start=start_time)

# Initialize the webcam
_, prev_frame = cap.read()

# Initialize model
model_path = "models/neuflow_mixed.onnx"
estimator = NeuFlowV2(model_path)

cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Estimate optical flow
    flow = estimator(prev_frame, frame)

    # Draw Flow
    flow_img = draw_flow(flow, prev_frame)
    prev_frame = frame

    cv2.imshow("Optical Flow", flow_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break