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

# Load the video
video = cv2.VideoCapture("./images/video_test4.mp4")
if (video.isOpened() == False): 
  print("Unable to read camera feed")

# Create output video
fps = 30
frame_width = int(video.get(3))
frame_height = int(video.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

# Configuration for text
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 255, 255)
thickness = 2

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

    # Find the person position on image
    position = getPersonPosition(frame, model_pose_estimation)

    # Add info text on the result frame
    outImg = position[0]
    outImg = cv2.putText(outImg, ("Posicao: " + str(position[1])), org, font, fontScale, color, thickness, cv2.LINE_AA)
    if(interestArea != None):
      outImg = cv2.putText(outImg, ("Objetos: " + ' '.join(interestArea[1])), (50, 100), font, 0.7, color, thickness, cv2.LINE_AA)
    cv2.imshow("output", outImg)
    out.write(outImg)

    # Save the results
    cv2.imwrite("yolo_output.png", np.squeeze(results.render()))
    cv2.imwrite("input.png", frame)
    cv2.imwrite("keypoints.png", position[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()