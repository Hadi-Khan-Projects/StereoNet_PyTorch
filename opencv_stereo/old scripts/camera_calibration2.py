import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

images_left = glob.glob('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_images/left/*.jpg')
images_right = glob.glob('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_images/right/*.jpg')

for fname in images_left:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpointsL.append(corners2)

# Calibration
ret, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, gray.shape[::-1], None, None)

np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/left.npy', mtxL)


for fname in images_right:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpointsR.append(corners2)

# Calibration
ret, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, gray.shape[::-1], None, None)

np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/right.npy', mtxR)

# Stereo Calibration
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, gray.shape[::-1])

# Generate Q matrix
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, gray.shape[::-1], R, T)

np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/Q_matrix.npy', Q)