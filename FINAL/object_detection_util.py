import cv2
import numpy as np
from PIL import Image

def isInside(x, y, area_x1, area_y1, area_x2, area_y2):
    return (x >= area_x1 and x <= area_x2) and (y >= area_y1 and y <= area_y2)

def getImageInterestArea(image, result):
    pd = result.pandas().xyxy[0]

    # Encontra area da pessoa com maior confianca
    person_x1, person_y1, person_x2, person_y2 = 0, 0, 0, 0
    conf = 0 
    for index, row in pd.iterrows():
        if row["class"] == 0 and row["confidence"] >= 0.5 and row["confidence"] >= conf:
            conf = row["confidence"]
            person_x1 = int(row["xmin"])
            person_y1 = int(row["ymin"])
            person_x2 = int(row["xmax"])
            person_y2 = int(row["ymax"])

    if(conf == 0):
        return None

    # Verifica todos os objetos na região da pessoa
    objects = []
    objects_x1 = [person_x1]
    objects_y1 = [person_y1]
    objects_x2 = [person_x2]
    objects_y2 = [person_y2]
    for index, row in pd.iterrows():
        x1 = int(row["xmin"])
        y1 = int(row["ymin"])
        x2 = int(row["xmax"])
        y2 = int(row["ymax"])
        heigth = y2  - y1
        width = x2 - x1

        # Se existe um ponto do objeto no interior da area de interesse
        if (isInside(x1, y1, person_x1, person_y1, person_x2, person_y2)):
            objects_x1.append(x1)
            objects_y1.append(y1)
            objects_x2.append(x2)
            objects_y2.append(y2)
            objects.append(row["name"])
        elif (isInside(x2, y2, person_x1, person_y1, person_x2, person_y2)):
            objects_x1.append(x1)
            objects_y1.append(y1)
            objects_x2.append(x2)
            objects_y2.append(y2)
            objects.append(row["name"])
        elif (isInside((x1+width), y1, person_x1, person_y1, person_x2, person_y2)):
            objects_x1.append(x1)
            objects_y1.append(y1)
            objects_x2.append(x2)
            objects_y2.append(y2)
            objects.append(row["name"])
        elif (isInside(x1, (y2+heigth), person_x1, person_y1, person_x2, person_y2)):
            objects_x1.append(x1)
            objects_y1.append(y1)
            objects_x2.append(x2)
            objects_y2.append(y2)
            objects.append(row["name"])

    # Recorta a area que possue todos os objetos de interesse
    cropped_image = image[min(objects_y1):max(objects_y2), min(objects_x1):max(objects_x2)]    
    return (cropped_image, objects)