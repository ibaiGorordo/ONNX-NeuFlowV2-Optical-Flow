from datetime import timedelta
import cv2
from cap_from_youtube import cap_from_youtube
from yolov10 import YOLOv10
from neuflowv2 import NeuFlowV2, draw_flow

# Initialize video
videoUrl = 'https://youtu.be/9gmsUF3wpGM?si=tzBfIxJ6ij29KAfJ'
start_time = timedelta(minutes=23, seconds=15)
cap = cap_from_youtube(videoUrl, start=start_time)

# Initialize the webcam
_, prev_frame = cap.read()

# Initialize model
model_path = "models/neuflow_mixed.onnx"
estimator = NeuFlowV2(model_path)

# Initialize object detector
model_path = "models/yolov10m.onnx"
detector = YOLOv10(model_path, conf_thres=0.2)

cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    class_ids, boxes, _ = detector(prev_frame)
    boxes = boxes[class_ids<10] # Only keep traffic related objects

    # Estimate optical flow
    flow = estimator(prev_frame, frame)

    # Draw Flow
    flow_img = draw_flow(flow, prev_frame, boxes)
    prev_frame = frame

    cv2.imshow("Optical Flow", flow_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break