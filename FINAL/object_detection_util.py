import cv2
import numpy as np
from PIL import Image

def getImageInterestArea(image, result):
    pd = result.pandas().xyxy[0]
    cropped_image = None
    # Encontra box 
    for index, row in pd.iterrows():
        if row["class"] == 0 and row["confidence"] >= 0.5: 
            x1 = int(row["xmin"])
            y1 = int(row["ymin"])
            x2 = int(row["xmax"])
            y2 = int(row["ymax"])
            cropped_image = image[y1:y2, x1:x2]
            cv2.imshow("croped", cropped_image)
    return (cropped_image, pd)


    # # Encontra area da pessoa
    # # person_x1, person_y1, person_x2, person_y2 = None, None, None, None 
    # # for index, row in pd.iterrows():
    # #     if row["class"] == 0 and row["confidence"] >= 0.5: 
    # #         person_x1 = int(row["xmin"])
    # #         person_y1 = int(row["ymin"])
    # #         person_x2 = int(row["xmax"])
    # #         person_y2 = int(row["ymax"])

    # for index, row in pd.iterrows():
    #     x1 = int(row["xmin"])
    #     y1 = int(row["ymin"])
    #     x2 = int(row["xmax"])
    #     y2 = int(row["ymax"])

    #     if (x1 <= person_x2 and x1 >= person_x1) and (y1 <= person_y2 and y1 >= person_y2):
    #         if(x2 > person_x2):
    #             person_x2 = x2
    #         if(y2 > person_y2):
    #             person_y2 = person_y2
    #         if(x2 )
    #         cropped_image = image[y1:y2, x1:x2]
    #         cv2.imshow("croped", cropped_image)
    # return (cropped_image, pd)
