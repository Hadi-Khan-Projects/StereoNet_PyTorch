import os
import time
import matplotlib.pyplot as plt
import cv2

import PIL.Image
import numpy as np
import torch
import torchvision.transforms

from src.stereonet.model import StereoNet
import src.stereonet.utils_io

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

# random test



# Open video capture for both webcams
# Adjust the indices if your webcams are different
left_video = cv2.VideoCapture(0) 
right_video = cv2.VideoCapture(1)

start = time.time()

while True:
    # Read frames from both webcams
    left_ret, left_frame = left_video.read()
    right_ret, right_frame = right_video.read()
    
    # If frames were successfully read
    if left_ret and right_ret:
        # Convert the frames to the correct format (np.uint8 and RGB)
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        
        images = [left_frame, right_frame]

        # For torch stack, images must have same height and width, so perform center crop
        min_height = min(images[0].shape[0], images[1].shape[0])
        min_width = min(images[0].shape[1], images[1].shape[1])

        # Put the images onto the GPU and in [Channel, Height, Width] order
        tensored = [torch.permute(torch.from_numpy(array).to(torch.float32).to(device), (2, 0, 1)) for array in images]

        print(tensored[0].shape)
        print(tensored[1].shape)

        # Zero mean/unit standard deviation normalize the images. specific alues from andrewlstewart train set
        normalizer = torchvision.transforms.Normalize((111.5684, 113.6528), (61.9625, 62.0313))
        normalized = [normalizer(img) for img in tensored]

        # Crop and stack image pair
        cropper = torchvision.transforms.CenterCrop((min_height, min_width))
        stack = torch.stack([cropper(img) for img in normalized], dim=0)  # Channel, Height, Width

        # Model takes in a batch, so add a dummy axis.
        batch = torch.unsqueeze(normalized, dim=0)  # Batch, Channel, Height, Width

        # Perform model inference in eval mode
        model.eval()
        with torch.no_grad():
            batched_prediction = model(batch)
            
        end = time.time()
        print(end - start)

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

        end = time.time()
        print(end - start)
    
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
left_video.release()
right_video.release()
# Destroy all the windows
cv2.destroyAllWindows()

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