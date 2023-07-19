import cv2
import time
import numpy as np

# Initialize parameters for stereo block matching
params = {'minDisparity': 0,
          'numDisparities': 112,
          'blockSize': 23,
          'P1': 255,
          'P2': 255,
          'disp12MaxDiff': 255,
          'uniquenessRatio': 0,
          'speckleWindowSize': 0,
          'speckleRange': 0,
          'preFilterCap': 0,
          'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY}

# Create a window to display results
cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

# Function to update the stereo block matcher when a trackbar moves
def on_trackbar_change(val, param):
    params[param] = val

# Create trackbars for parameters
for param in params:
    cv2.createTrackbar(param, 'Disparity', params[param], 255, lambda val, param=param: on_trackbar_change(val, param))

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('opencv_stereo/calibration_data/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open video capture for both webcams
# Adjust the indices if your webcams are different
left_video = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
right_video = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Define the codec and create a VideoWriter object for left camera
fourcc = cv2.VideoWriter_fourcc(*'XVID')
left_out = cv2.VideoWriter('saved_videos/left_camera.avi', fourcc, 20.0, (640, 480))  # Adjust frame rate and resolution as needed

# Define the codec and create a VideoWriter object for disparity
disparity_out = cv2.VideoWriter('saved_videos/disparity.avi', fourcc, 20.0, (640, 480))  # Adjust frame rate and resolution as needed

while True:
    start = time.time()

    # Read frames from both webcams
    left_ret, left_frame = left_video.read()
    right_ret, right_frame = right_video.read()

    # If frames were successfully read
    if left_ret and right_ret:
        # Undistort and rectify images
        left_frame_rectified = cv2.remap(left_frame, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        right_frame_rectified = cv2.remap(right_frame, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)         

        # Convert both frames to grayscale
        left_gray = cv2.cvtColor(left_frame_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame_rectified, cv2.COLOR_BGR2GRAY)

        # Create a stereo block matcher with current parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity=params['minDisparity'],
            numDisparities=params['numDisparities'],
            blockSize=params['blockSize'],
            P1=params['P1'],
            P2=params['P2'],
            disp12MaxDiff=params['disp12MaxDiff'],
            uniquenessRatio=params['uniquenessRatio'],
            speckleWindowSize=params['speckleWindowSize'],
            speckleRange=params['speckleRange'],
            preFilterCap=params['preFilterCap'],
            mode=params['mode']
        )

        # Calculate disparity (depth map)
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Normalize disparity to the range 0-255 for viewing
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert grayscale to BGR
        disparity_color = cv2.cvtColor(disparity, cv2.COLOR_GRAY2BGR)

        # Write the rectified frame and disparity to the files
        left_out.write(left_frame_rectified)
        disparity_out.write(disparity_color )

        # Display disparity
        cv2.imshow('Disparity', disparity)
        cv2.imshow('Left Camera', left_frame_rectified)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Print frames per second
        print("FPS: ", 1.0 / (time.time() - start))

# Release the VideoWriter and VideoCapture objects
left_out.release()
disparity_out.release()
left_video.release()
right_video.release()

cv2.destroyAllWindows()