import cv2
import torch
from torch import load
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler, create_detection_live_display
import sys
import time 


# Initialize logger and handlers
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

transforms = A.Compose(
        [   
            A.Resize(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
    )

model = DETR(num_classes=4)
model.eval()
model.load_pretrained('checkpoints/99_model.pt')
CLASSES = get_classes() 
COLORS = get_colors() 

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame from camera")
        break
    # Mirror the camera frame horizontally
    frame = cv2.flip(frame, 1)
        
    # Time the inference
    inference_start = time.time()
    transformed = transforms(image=frame)
    result = model(torch.unsqueeze(transformed['image'], dim=0))
    inference_time = (time.time() - inference_start) * 1000  # Convert to ms

    probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > 0.5  # Set IoU threshold

    batch_indices, query_indices = torch.where(keep_mask)

    # Limit to max_det = 1 (top confidence)
    if len(query_indices) > 0:
        top_idx = max_probs[batch_indices, query_indices].argmax()
        batch_indices = batch_indices[top_idx:top_idx+1]
        query_indices = query_indices[top_idx:top_idx+1]

    height, width = frame.shape[:2]
    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (width, height))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    # Prepare detection results for logging
    detections = []
    for bclass, bprob, bbox in zip(classes, probas, bboxes): 
        bclass_idx = bclass.detach().numpy()
        bprob_val = bprob.detach().numpy() 
        x1, y1, x2, y2 = map(int, bbox.detach().numpy())
        color = COLORS[bclass_idx]

        # Draw a thinner bounding box
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Prepare label text
        label = f"{CLASSES[bclass_idx]}: {round(float(bprob_val), 2)}"

        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # Draw filled rectangle for label background (with some transparency)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        alpha = 0.6  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Put label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Calculate FPS
    frame_count += 1
    if frame_count % 30 == 0:  # Log every 30 frames
        elapsed_time = time.time() - fps_start_time
        fps = 30 / elapsed_time
        
        # Log detection results and performance
        if detections:
            detection_handler.log_detections(detections, frame_id=frame_count)
        detection_handler.log_inference_time(inference_time, fps)
        
        # Reset FPS counter
        fps_start_time = time.time()

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        logger.realtime("Stopping real-time detection...")
        break

cap.release() 
cv2.destroyAllWindows()
