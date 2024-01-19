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
        
        masks = results[0].masks.data
        
        # print(results[0].masks)
        # break
        
        selected_masks = get_selected_masks((340, 180), masks)
        
        if(len(selected_masks) != 0):
            mask_union = torch.clamp(sum(selected_masks), 0, 1)
            mask_union = mask_union.numpy()
            mask_union = (mask_union * 255).astype(np.uint8)
            
            cv2.imshow("Test", mask_union)
        else:
            print("No object is selected!")
     
        end = time.perf_counter()
        total_time = end - start
        fps = 1/total_time
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.putText(img = annotated_frame, text = f"FPS: {int(fps)}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (0, 0, 255), thickness = 4, lineType = cv2.LINE_AA)
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        #Break the loop if the end of the video is reached
        break
    
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
        