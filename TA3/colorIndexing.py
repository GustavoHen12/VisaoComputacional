from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import glob

PATH_OBJECTS = glob.glob('./objetos/*.png')
PATH_SCENES = glob.glob('./cenas/*.png')
cv.namedWindow('object', cv.WINDOW_NORMAL)
cv.resizeWindow('object', 250,250)

cv.namedWindow('scene', cv.WINDOW_NORMAL)
cv.resizeWindow('scene', 600,400)

# Carregar os modelos
objects_histogram = []
objects_image = []
for image_path in PATH_OBJECTS:
    img = cv.imread(image_path)
    hsv_image = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    # Para cada modelo criar histograma
    hist = cv.calcHist([hsv_image],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    objects_histogram.append(hist)
    objects_image.append(img)

# Carregar cenas
for scene_path in PATH_SCENES:
    img = cv.imread(scene_path)
    hsv_image = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    
    # Cria histograma da imagem
    hist = cv.calcHist([hsv_image],[0, 1], None, [180, 256], [0, 180, 0, 256] )

    # Comparar cenas com histograma
    for j in range(0, len(objects_histogram)):
        obj_hist = objects_histogram[j] 
        obj_img = objects_image[j]

        R = obj_hist/hist
        h,s,v = cv.split(hsv_image)
        B = R[h.ravel(),s.ravel()]
        B = np.minimum(B,1)
        B = B.reshape(hsv_image.shape[:2])

        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        cv.filter2D(B,-1,disc,B)
        B = np.uint8(B)
        cv.normalize(B,B,0,255,cv.NORM_MINMAX)

        ret,thresh = cv.threshold(B,50,255,0)
        thresh = cv.merge((thresh,thresh,thresh))
        res = cv.bitwise_and(img,thresh)
        res = np.hstack((img,thresh,res))

        cv.imshow('scene', res)
        cv.imshow('object', obj_img)
        cv.waitKey(500)  
