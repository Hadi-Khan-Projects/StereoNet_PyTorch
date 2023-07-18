import os
import time
import matplotlib.pyplot as plt
import cv2

import PIL.Image
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms

from src.stereonet.model import StereoNet
import src.stereonet.utils_io

def depth_to_pcd(depth, color_img):
    # Here, we assume that depth is a single-channel grayscale image, and color_img is a 3-channel color image.
    # Both depth and color_img should be the same size.

    # Adjust these parameters as needed.
    fx, fy = 320, 320  # focal lengths
    cx, cy = 319.5, 239.5  # optical center

    # Generate the x, y coordinates
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Back-project to 3D (assuming metric depth)
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([x3.flat, y3.flat, z3.flat]))

    # If you have a color image for the points, assign it here
    pcd.colors = o3d.utility.Vector3dVector(color_img.reshape(-1, 3) / 255.0)

    return pcd

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Initial empty point cloud
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# Check for CUDA availability and set device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    print("CUDA is not available. The model will use CPU instead, which will be much slower")
    device = 'cpu'

# Define the path to the checkpoint
checkpoint_path = "C:/Users/User/Desktop/StereoNet/StereoNet_PyTorch/model_checkpoint/epoch=21-step=696366.ckpt"

# Load the model from the trained checkpoint
model = StereoNet.load_from_checkpoint(checkpoint_path)

# Move the model to your device
model = model.to(device)

# Define the paths to the left and right video files
left_video_path = "C:/Users/User/Desktop/Work/Uni/TanerFootage/Test1.mp4"
right_video_path = "C:/Users/User/Desktop/Work/Uni/TanerFootage/Test0.mp4"

# Open video capture for both videos
left_video = cv2.VideoCapture(left_video_path) 
right_video = cv2.VideoCapture(right_video_path)

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Initial empty point cloud
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

vis.get_view_control().set_zoom(0.5)
vis.get_view_control().set_lookat([0,0,0])

while True:
    start = time.time()

    # Read frames from both webcams
    left_ret, left_frame_col = left_video.read()
    right_ret, right_frame_col = right_video.read()
    
    # If frames were successfully read
    if left_ret and right_ret:
        # Convert the frames to the correct format (np.uint8 and RGB)
        left_frame = np.expand_dims(cv2.cvtColor(left_frame_col, cv2.COLOR_RGB2GRAY).astype(np.uint8), axis=-1)
        right_frame = np.expand_dims(cv2.cvtColor(right_frame_col, cv2.COLOR_RGB2GRAY).astype(np.uint8), axis=-1)
        
        images = [left_frame, right_frame]

        # For torch stack, images must have same height and width, so perform center crop
        min_height = min(images[0].shape[0], images[1].shape[0])
        min_width = min(images[0].shape[1], images[1].shape[1])

        # Put the images onto the GPU and in [Channel, Height, Width] order
        tensored = [torch.permute(torch.from_numpy(array).to(torch.float32).to(device), (2, 0, 1)) for array in images]

        # Crop and stack image pair
        cropper = torchvision.transforms.CenterCrop((min_height, min_width))
        stack = torch.concatenate(list(map(cropper, tensored)), dim=0)  # C, H, W

        # Zero mean/unit standard deviation normalize the images. specific alues from andrewlstewart train set
        normalizer = torchvision.transforms.Normalize((111.5684, 113.6528), (61.9625, 62.0313))
        normalized = normalizer(stack)

        # Model takes in a batch, so add a dummy axis.
        batch = torch.unsqueeze(normalized, dim=0)  # Batch, Channel, Height, Width

        # Perform model inference in eval mode
        model.eval()
        with torch.no_grad():
            batched_prediction = model(batch)
            
        end = time.time()
        #print("This frame took: ", end - start)

        # Do some partial crops on the disparity predictions (most egregious errors are caused from disparities being not well defined on the left/right borders
        crops = (1.0, 0.95, 0.9, 0.85)
        predictions_torch = []
        for crop_percentage in crops:

            cropper = torchvision.transforms.CenterCrop((round(min_height), round(min_width*crop_percentage)))
            predictions_torch.append(cropper(batched_prediction[0]))

        # Convert from torch tensors on GPU to numpy arrays on CPU and switch back to channels last notation
        predictions = [np.moveaxis(pred.detach().cpu().numpy(), 0, 2) for pred in predictions_torch]

        # Visalize the (left/right) image pair and the disparity maps for the various crops. 
        # These disparity maps have been rescaled to [0, 255] so that they are easily viewed as images.
        resized_predictions = []
        for prediction in predictions:
            prediction = PIL.Image.fromarray(((prediction-prediction.min())/(prediction.max()-prediction.min())*255).astype(np.uint8)[..., 0])
            prediction = prediction.resize((images[0].shape[1], images[0].shape[0]))
            resized_predictions.append(np.array(prediction))
    
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Convert the depth map to a point cloud
        arr = np.array(prediction)
        pcd_new = depth_to_pcd(arr, left_frame_col)
        
        # Update the point cloud points
        pcd.points = pcd_new.points
        pcd.colors = pcd_new.colors

        #print(np.asarray(pcd.points))
        #print("0-----0")
        #print(np.asarray(pcd.colors))

        print("Number of points: ", len(pcd.points))
        print("Min coords: ", np.min(np.asarray(pcd.points), axis=0))
        print("Max coords: ", np.max(np.asarray(pcd.points), axis=0))

        # Update the visualizer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        o3d.io.write_point_cloud("test.ply", pcd)


        break
vis.destroy_window()

fig, ax = plt.subplots(nrows=2, ncols=len(predictions), figsize=(9, 7))
ax[0, 0].imshow(images[0], cmap='gray')
ax[0, 0].set_title("Left image")
ax[0, 1].imshow(images[1], cmap='gray')
ax[0, 1].set_title("Right image")
for idx, (prediction, crop) in enumerate(zip(resized_predictions, crops)):
    ax[1, idx].imshow(prediction, cmap='gray')
    ax[1, idx].set_title(f"Left disparity map\n(crop={crop})")
[axi.set_axis_off() for axi in ax.ravel()]
plt.tight_layout()

# Show plot lmao
plt.show()

# After the loop release the cap object
left_video.release()
right_video.release()
# Destroy all the windows
cv2.destroyAllWindows()
