import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
# https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#force-reload
# https://github.com/ultralytics/yolov5/issues/9115

# Define the path to the YOLOv5 directory
yolov5_dir = Path("./yolov5")

# Load the YOLOv5 model
model = torch.hub.load(str(yolov5_dir), "custom", path=str(yolov5_dir / "yolov5s.pt"), source="local")
model.conf = 0.4  # Set the confidence threshold for detection

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

cv2.namedWindow('croped', cv2.WINDOW_NORMAL)
cv2.namedWindow('yolo_output', cv2.WINDOW_NORMAL)

# Load the video
video = cv2.VideoCapture(0)
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Convert the frame to PIL Image format
    frame_pil = Image.fromarray(frame[:, :, ::-1])

    # Perform object detection using YOLOv5
    results = model(frame_pil, size=640)
    # pd = results.pandas().xyxy[0]
    # for index, row in pd.iterrows():
    #     if row["class"] == 0 and row["confidence"] >= 0.5: 
    #         x1 = int(row["xmin"])
    #         y1 = int(row["ymin"])
    #         x2 = int(row["xmax"])
    #         y2 = int(row["ymax"])
    #         cropped_image = frame[y1:y2, x1:x2]
    #         cv2.imshow("croped", cropped_image)

    # Display the results
    cv2.imshow("yolo_output", np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
