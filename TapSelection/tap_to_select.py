import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_selected_masks(idx_pt, masks): #Returns a list of masks
    selected_masks = []
    for mask in masks:
        if mask[idx_pt[0]][idx_pt[1]] != 0:
            selected_masks.append(mask)
    return selected_masks


def highlight_mask(image, mask):
    
    # Create a copy of the original image
    highlighted_image = image.copy()

    # Set the masked portion to a translucent red color
    highlighted_image[mask != 0] = [0, 0, 255]  # Set the BGR values for red

    # Blend the original image with the highlighted mask using addWeighted
    alpha = 0.5  # Adjust the alpha value for the level of transparency
    final_image = cv2.addWeighted(image, 1 - alpha, highlighted_image, alpha, 0)

    return final_image


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Load the YOLOv8 model
model = YOLO('yolov8s-seg.pt')

# Open the video file
video_path = "./objects-fixed-cam.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        start = time.perf_counter()
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        end = time.perf_counter()
        total_time = end - start
        fps = 1/total_time
        
        masks = results[0].masks.data
        
        selected_masks = get_selected_masks((340, 180), masks)
        
        if(len(selected_masks) != 0):
            mask_union = torch.clamp(sum(selected_masks), 0, 1)
            mask_union = mask_union.numpy()
            mask_union = (mask_union * 255).astype(np.uint8)
            mask_union = cv2.resize(mask_union, (frame.shape[1], frame.shape[0]))
            mask_union = np.clip(mask_union, a_min = 0, a_max = 255) 
            cv2.imshow("Selected Masks", mask_union)
        else:
            print("No object is selected!")
        
        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # # # Display the annotated frame
        # cv2.putText(img = annotated_frame, text = f"FPS: {int(fps)}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (0, 0, 255), thickness = 4, lineType = cv2.LINE_AA)
        # cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        seg_frame = highlight_mask(frame, mask_union)
        cv2.putText(img = seg_frame, text = f"FPS: {int(fps)}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (0, 0, 255), thickness = 4, lineType = cv2.LINE_AA)
        cv2.imshow("Segmentation Result", seg_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        #Break the loop if the end of the video is reached
        break
    
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
        