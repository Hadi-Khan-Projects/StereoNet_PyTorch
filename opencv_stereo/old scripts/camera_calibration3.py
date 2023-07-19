import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# prepare object points, multiply by the size of the squares
objp = objp * 20

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imagesLeft = glob.glob('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_images/left/*.jpg')
imagesRight = glob.glob('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_images/right/*.jpg')

for fname in zip(imagesLeft, imagesRight):

    imgL = cv2.imread(fname[0])
    imgR = cv2.imread(fname[1])
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (7,9), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (7,9), None)

    # If found, add object points, image points (after refining them)
    if retL and retR:
        objpoints.append(objp)

        corners2L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(corners2L)

        corners2R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(corners2R)

# Camera calibration
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

# Stereo calibration
_, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1])

# Save calibration data
np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_data/mtxL.npy', mtxL)
np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_data/mtxR.npy', mtxR)
np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_data/distL.npy', distL)
np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_data/distR.npy', distR)
np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_data/R.npy', R)
np.save('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/cv2_implementation/calib_data/T.npy', T)
