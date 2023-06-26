import torchvision
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
import math

# create the list of keypoints.

def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    # initialize a set of colors from the rainbow spectrum
    cmap = plt.get_cmap('rainbow')
    # create a copy of the image
    img_copy = img.copy()
    # pick a set of N color-ids from the spectrum
    color_id = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
    # iterate for every person detected
    for person_id in range(len(all_keypoints)):
      # check the confidence score of the detected person
      if confs[person_id]>conf_threshold:
        # grab the keypoint-locations for the detected person
        keypoints = all_keypoints[person_id, ...]
        # grab the keypoint-scores for the keypoints
        scores = all_scores[person_id, ...]
        # iterate for every keypoint-score
        for kp in range(len(scores)):
            # check the confidence score of detected keypoint
            if scores[kp]>keypoint_threshold:
                # convert the keypoint float-array to a python-list of intergers
                keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                # pick the color at the specific color-id
                color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                # draw a cirle over the keypoint location
                cv2.circle(img_copy, keypoint, 5, color, -1)

    return img_copy

def calcular_angulos(hx, hy, sx, sy, kx, ky):
    sk = math.sqrt((sx - kx) ** 2 + (sy - ky) ** 2)
    hk = math.sqrt((hx - kx) ** 2 + (hy - ky) ** 2)
    hs = math.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)

    if sk < hs:
       return "sentado"
    
    angulo_a = math.degrees(math.acos((hk ** 2 + hs ** 2 - sk ** 2) / (2 * hk * hs)))
    angulo_b = math.degrees(math.acos((sk ** 2 + hs ** 2 - hk ** 2) / (2 * sk * hs)))
    angulo_c = math.degrees(math.acos((sk ** 2 + hk ** 2 - hs ** 2) / (2 * sk * hk)))

    if int(angulo_b) <= 60:
       return "sentado"

    return "em pe"

def get_pose_estimated(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    keypoints_index = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee', 'right_knee', 'left_ankle','right_ankle']
    
    ## Tenta formar um triangulo entre o ombro, bacia e joelho
    for person_id in range(len(all_keypoints)):
      if confs[person_id]>conf_threshold:
        keypoints = all_keypoints[person_id, ...]
        scores = all_scores[person_id, ...]
        print(keypoints)
        print("Right:")
        right_hip = keypoints[keypoints_index.index('right_hip'), :2].detach().numpy().astype(np.int32)
        right_shoulder = keypoints[keypoints_index.index('right_shoulder'), :2].detach().numpy().astype(np.int32)
        right_knee = keypoints[keypoints_index.index('right_knee'), :2].detach().numpy().astype(np.int32)
        print(calcular_angulos(right_hip[0], right_hip[1], right_shoulder[0], right_shoulder[1], right_knee[0], right_knee[1]))    

        print("Left:")
        left_hip = keypoints[keypoints_index.index('left_hip'), :2].detach().numpy().astype(np.int32)
        left_shoulder = keypoints[keypoints_index.index('left_shoulder'), :2].detach().numpy().astype(np.int32)
        left_knee = keypoints[keypoints_index.index('left_knee'), :2].detach().numpy().astype(np.int32)
        print(calcular_angulos(left_hip[0], left_hip[1], left_shoulder[0], left_shoulder[1], left_knee[0], left_knee[1]))    

    return "sentado"


def getPersonPosition(img, model):
   transform = T.Compose([T.ToTensor()])
   img_tensor = transform(img)

   # forward-pass the model
   # the input is a list, hence the output will also be a list
   output = model([img_tensor])[0]
   
   keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
   cv2.imshow("points", keypoints_img)
   pose_estimated = get_pose_estimated(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)

   return (pose_estimated, keypoints_img)
