import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (20,0,0), (40,0,0) ....,(160,120,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * 20

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imagesL = glob.glob('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/camera_calibrator/calib_images/left/*.jpg')
imagesR = glob.glob('C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/camera_calibrator/calib_images/right/*.jpg')

img_shape = None

for fnameL, fnameR in zip(imagesL, imagesR):
    imgL = cv2.imread(fnameL)
    imgR = cv2.imread(fnameR)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (8,6), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (8,6), None)

    # If found, add object points, image points (after refining them)
    if retL == True and retR == True:
        objpoints.append(objp)

        cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

    if img_shape is None:
        img_shape = grayR.shape[::-1]

# Now we can calibrate each camera and then calibrate them together
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, img_shape, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, img_shape, None, None)

# Now we are going to stereo calibrate
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camera matrices so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                        cv2.TERM_CRITERIA_EPS, 100, 1e-5)
retStereo, new_mtxL, distL, new_mtxR, distR, rot, trans, E, F = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, img_shape, criteria = stereocalib_criteria, flags = flags)

# Then, we rectify the distortions
rectify_scale = 0  # if 0, we rectify the scaled images, if 1 we keep the original resolution
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, img_shape, rot, trans, rectify_scale, (0,0))  # last parameter is alpha, if 0= cropped, if 1= not cropped
left_map1, left_map2 = cv2.initUndistortRectifyMap(new_mtxL, distL, R1, P1, img_shape, cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the program to work faster
right_map1, right_map2 = cv2.initUndistortRectifyMap(new_mtxR, distR, R2, P2, img_shape, cv2.CV_16SC2)

# Now you can save the needed matrices for later use
np.save('Q_matrix.npy', Q)
np.save('left_map.npy', [left_map1, left_map2])
np.save('right_map.npy', [right_map1, right_map2])