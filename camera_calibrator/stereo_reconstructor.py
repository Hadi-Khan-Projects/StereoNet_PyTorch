import numpy as np
import cv2

# Initialize the two cameras
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# Set up the stereo camera parameters
# Here, we are using a StereoBM which is a basic and fast method to calculate disparity map
# More advanced methods such as StereoSGBM or deep learning methods could be used for better results
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Load calibration parameters
# These are placeholders and you should replace them with your own calibration parameters
Q = np.load('Q_matrix.npy') # The disparity-to-depth mapping matrix
left_map = np.load('left_map.npy') # The left lens undistortion mapping
right_map = np.load('right_map.npy') # The right lens undistortion mapping

while(True):
    # Capture frame-by-frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Rectify the images
    frame1_rect = cv2.remap(frame1, left_map[0], left_map[1], cv2.INTER_LINEAR)
    frame2_rect = cv2.remap(frame2, right_map[0], right_map[1], cv2.INTER_LINEAR)

    # Compute disparity map
    disparity = stereo.compute(cv2.cvtColor(frame1_rect, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2_rect, cv2.COLOR_BGR2GRAY))

    # Reconstruct 3D points
    points = cv2.reprojectImageTo3D(disparity, Q)

    # Display the resulting frame
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)
    cv2.imshow('disparity', disparity)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
