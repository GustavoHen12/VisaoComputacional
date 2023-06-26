import cv2
import numpy as np
from PIL import Image

def getImageInterestArea(image, result):
    pd = result.pandas().xyxy[0]
    for index, row in pd.iterrows():
        if row["class"] == 0 and row["confidence"] >= 0.5: 
            x1 = int(row["xmin"])
            y1 = int(row["ymin"])
            x2 = int(row["xmax"])
            y2 = int(row["ymax"])
            cropped_image = image[y1:y2, x1:x2]
            cv2.imshow("croped", cropped_image)
