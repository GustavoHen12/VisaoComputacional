import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from PIL import Image
from object_detection_util import getImageInterestArea
from pose_estimation_util import getPersonPosition

# Object detection model
# Define the path to the YOLOv5 directory
yolov5_dir = Path("./yolov5")

# Load the YOLOv5 model for object detection
model = torch.hub.load(str(yolov5_dir), "custom", path=str(yolov5_dir / "yolov5s.pt"), source="local")
model.conf = 0.4

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Pose estimation model
# create a model object from the keypointrcnn_resnet50_fpn class
model_pose_estimation = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model_pose_estimation.eval()

# img = cv2.imread('./images/jsent.png')
# getPersonPosition(img, model_pose_estimation)
# cv2.waitKey(500)

# Load the video
video = cv2.VideoCapture("./images/video_test4.mp4")
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    frame_pil = Image.fromarray(frame[:, :, ::-1])

    # Perform object detection using YOLOv5
    results = model(frame_pil, size=640)

    # Process area in the image with a person
    interestArea = getImageInterestArea(frame, results)

    if (interestArea != None and (interestArea[0]).all() != None):
        cv2.imshow("region_of_interest", interestArea[0])
        print(interestArea[1])
        position = getPersonPosition(interestArea[0], model_pose_estimation)
        
    
    # Display the results
    cv2.imshow("yolo_output", np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()