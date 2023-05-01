import numpy as np
import cv2 as cv
import glob

CHECKERBOARD = (7,7)
# CHECKERBOARD = (7,5)

image_height = 0
image_width = 0

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objpoints = [] 
imgpoints = []

directory = './Teste3/'

images = glob.glob(directory + '*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    image_height, image_width = img.shape[:2]

    # Busca os cantos do tabuleiro
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    # Se for encontrado, refina os pontos e em imgpoints
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        
        # Exibe resultado
        cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv.imwrite('resultado_parcial.png', img)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Calcula matriz da camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


# Calcula nova matriz e aplica na imagem
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (image_width,image_height), 1, (image_width,image_height))

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Desfaz distorção
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    # Corta a imagem e salva resultado
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite((fname + '_resultado_' + '.png'), dst)