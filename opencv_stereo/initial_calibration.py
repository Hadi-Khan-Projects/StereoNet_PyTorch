import cv2

# Open video capture for both webcams
# Adjust the indices if your webcams are different
left_video = cv2.VideoCapture(1)
right_video = cv2.VideoCapture(0)

image_num = 0

while left_video.isOpened():

    left_read, left_image = left_video.read()
    right_read, right_image = right_video.read()

    key = cv2.waitKey(5)
    if key == ord('s'): # wait for 's' key to save image
        cv2.imwrite('calibration_images/left/imageL' + str(image_num) + '.png', left_image)
        cv2.imwrite('calibration_images/right/imageR' + str(image_num) + '.png', right_image)
        print("images saved!")
        image_num += 1
    elif key == ord('q'): # Break loop if 'q' is pressed
        break

    cv2.imshow('Left Camera', left_image)
    cv2.imshow('Right Camera', right_image)

# When everything done, release the video capture and close windows
left_video.release()
right_video.release()
cv2.destroyAllWindows()