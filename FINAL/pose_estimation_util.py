import torchvision
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
import math

from pose_estimation import draw_skeleton_per_person

keypoints_index = ['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow', 'right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee', 'right_knee', 'left_ankle','right_ankle']

def encontrou_pontos(hx, hy, sx, sy, kx, ky):
    return not(sx == -1 or sy == -1 or hy == -1 or hx == -1 or kx == 1 or ky == -1)

def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    cmap = plt.get_cmap('rainbow')
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
            if scores[kp]>keypoint_threshold:
                keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                color = None
                if(kp == keypoints_index.index('right_knee') or kp == keypoints_index.index('left_knee')):
                    color = (0, 0, 255)
                elif (kp == keypoints_index.index('right_shoulder') or kp == keypoints_index.index('left_shoulder')):
                    color = (0, 255, 0)
                elif (kp == keypoints_index.index('right_hip') or kp == keypoints_index.index('left_hip')):
                    color = (255, 0, 0)
                else:
                    # color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                    color = (255, 255, 255)
                cv2.circle(img_copy, keypoint, 5, color, -1)
                # cv2.drawContours(img_copy, np.array[(right_hip, right_knee)], 0, (255,255,255), 2)

    keypoints = all_keypoints[person_id, ...]
    scores = all_scores[person_id, ...]
    right_hip = getKeypoint(keypoints, scores, 'right_hip', keypoint_threshold)
    right_shoulder = getKeypoint(keypoints, scores, 'right_shoulder', keypoint_threshold)
    right_knee = getKeypoint(keypoints, scores, 'right_knee', keypoint_threshold)

    left_hip = getKeypoint(keypoints, scores, 'left_hip', keypoint_threshold)
    left_shoulder = getKeypoint(keypoints, scores, 'left_shoulder', keypoint_threshold)
    left_knee = getKeypoint(keypoints, scores, 'left_knee', keypoint_threshold)
    
    if(encontrou_pontos(getMedia(left_hip[0], right_hip[0]), getMedia(left_hip[1], right_hip[1]),getMedia(left_shoulder[0], right_shoulder[0]),getMedia(left_shoulder[1], right_shoulder[1]),getMedia(left_knee[0], right_knee[0]),getMedia(left_knee[1], right_knee[1]))):
        # cv2.line(img_copy, (getMedia(right_hip[0], left_hip[0]),getMedia(right_hip[1], left_hip[1])), (getMedia(right_shoulder[0], left_shoulder[0]),getMedia(right_shoulder[1], left_shoulder[1])), (255,255,255), 2)
        cv2.line(img_copy, (int(getMedia(right_hip[0], left_hip[0])), int(getMedia(right_hip[1], left_hip[1]))), (int(getMedia(right_shoulder[0], left_shoulder[0])),int(getMedia(right_shoulder[1], left_shoulder[1]))), (255,255,255), 3)
        cv2.line(img_copy, (int(getMedia(right_hip[0], left_hip[0])), int(getMedia(right_hip[1], left_hip[1]))), (int(getMedia(right_knee[0], left_knee[0])),int(getMedia(right_knee[1], left_knee[1]))), (255,255,255), 3)
        cv2.line(img_copy, (int(getMedia(right_shoulder[0], left_shoulder[0])), int(getMedia(right_shoulder[1], left_shoulder[1]))), (int(getMedia(right_knee[0], left_knee[0])),int(getMedia(right_knee[1], left_knee[1]))), (255,255,255), 3)
    return img_copy


def calcular_angulos(img, hx, hy, sx, sy, kx, ky):
    img_copy = img.copy()
    ph = (hx, hy)
    ps = (sx, sy)
    pk = (kx, ky)
    print(ph,ps,pk)
    # print(hx, hy, sx, sy, kx, ky)

    if(sx == -1 or sy == -1 or hy == -1 or hx == -1 or kx == 1 or ky == -1):
       return "Undefined"

    # if(kx == -1 or ky == -1):
    #     return "sentado"
    # if(sx == -1 or sy == -1 or hy == -1 or hx == -1):
    #     return "em pe"

    # Calculando tamanho dos lados
    sk = math.sqrt((sx - kx) ** 2 + (sy - ky) ** 2)
    hk = math.sqrt((hx - kx) ** 2 + (hy - ky) ** 2)
    hs = math.sqrt((hx - sx) ** 2 + (hy - sy) ** 2)

    print(sk, hk, hs)
    
    # print((hk * 2 + hs * 2 - sk ** 2) / (2 * hk * hs))
    # print((sk * 2 + hs * 2 - hk ** 2) / (2 * sk * hs))
    # print((sk * 2 + hk * 2 - hs ** 2) / (2 * sk * hk))

    rad_a = math.acos((hk ** 2 + hs ** 2 - sk ** 2) / (2 * hk * hs))

    angulo_a = math.degrees(rad_a)
    # angulo_b = math.degrees(math.acos((sk ** 2 + hs ** 2 - hk ** 2) / (2 * sk * hs)))
    # angulo_c = math.degrees(math.acos((sk ** 2 + hk ** 2 - hs ** 2) / (2 * sk * hk)))

    # print(angulo_a, angulo_b, angulo_c)
    print(angulo_a)

    if int(angulo_a) <= 140:
       return "sentado"

    return "em pe"

def getKeypoint(keypoints, scores, point, keypoint_threshold=2):
    keypoint = keypoints[keypoints_index.index(point), :2].detach().numpy().astype(np.int32)
    score = scores[keypoints_index.index(point)]
    if score > keypoint_threshold:
        return keypoint
    return (-1, -1)

def get_pose_estimated(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    
    result = "Undefined"
    ## Tenta formar um triangulo entre o ombro, bacia e joelho
    for person_id in range(len(all_keypoints)):
      if confs[person_id]>conf_threshold:
        keypoints = all_keypoints[person_id, ...]
        scores = all_scores[person_id, ...]
        # print(keypoints)
        # print("Right:")
        # right_hip = getKeypoint(keypoints, scores, 'right_hip', keypoint_threshold)
        # right_shoulder = getKeypoint(keypoints, scores, 'right_shoulder', keypoint_threshold)
        # right_knee = getKeypoint(keypoints, scores, 'right_knee', keypoint_threshold)
        # print(calcular_angulos(img, right_hip[0], right_hip[1], right_shoulder[0], right_shoulder[1], right_knee[0], right_knee[1]))

        # print("Left:")
        # left_hip = getKeypoint(keypoints, scores, 'left_hip', keypoint_threshold)
        # left_shoulder = getKeypoint(keypoints, scores, 'left_shoulder', keypoint_threshold)
        # left_knee = getKeypoint(keypoints, scores, 'left_knee', keypoint_threshold)
        # print(calcular_angulos(img, left_hip[0], left_hip[1], left_shoulder[0], left_shoulder[1], left_knee[0], left_knee[1]))

        right_hip = getKeypoint(keypoints, scores, 'right_hip', keypoint_threshold)
        right_shoulder = getKeypoint(keypoints, scores, 'right_shoulder', keypoint_threshold)
        right_knee = getKeypoint(keypoints, scores, 'right_knee', keypoint_threshold)

        left_hip = getKeypoint(keypoints, scores, 'left_hip', keypoint_threshold)
        left_shoulder = getKeypoint(keypoints, scores, 'left_shoulder', keypoint_threshold)
        left_knee = getKeypoint(keypoints, scores, 'left_knee', keypoint_threshold)
        result = calcular_angulos(img, getMedia(left_hip[0], right_hip[0]), getMedia(left_hip[1], right_hip[1]),getMedia(left_shoulder[0], right_shoulder[0]),getMedia(left_shoulder[1], right_shoulder[1]),getMedia(left_knee[0], right_knee[0]),getMedia(left_knee[1], right_knee[1]))
        print(result)    

    return result

def getMedia(a, b):
   return (a + b) / 2
 

def getPersonPosition(img, model):
   transform = T.Compose([T.ToTensor()])
   img_tensor = transform(img)

   # forward-pass the model
   # the input is a list, hence the output will also be a list
   output = model([img_tensor])[0]
   
   keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
   pose_estimated = get_pose_estimated(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)
   
   cv2.imshow("points", keypoints_img)
   return (keypoints_img, pose_estimated)