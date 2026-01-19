import cv2
import torch
from torch import load
from torchvision.ops import nms
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler, create_detection_live_display
import sys
import time 
import numpy as np
from collections import deque


# Initialize logger and handlers
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

# Setup device - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.realtime(f"Using device: {device}")

# ============== CONFIGURATION ==============
CONFIDENCE_THRESHOLD = 0.65  # Calibrated confidence threshold
NMS_THRESHOLD = 0.3          # Non-max suppression IoU threshold
TEMPORAL_WINDOW = 5          # Number of frames for temporal smoothing
MIN_DETECTION_FRAMES = 3     # Minimum frames to confirm detection
# ===========================================

# Use same transforms as training (normalized behavior)
transforms = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2()
    ]
)

CLASSES = get_classes()
COLORS = get_colors()
num_classes = len(CLASSES)

model = DETR(num_classes=num_classes)
model.load_pretrained('checkpoints/99_model.pt')
model = model.to(device)
model.eval()

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()

# Temporal smoothing: store recent detections
detection_history = deque(maxlen=TEMPORAL_WINDOW)


def apply_nms(boxes, scores, classes, iou_threshold=0.3):
    """Apply Non-Maximum Suppression to reduce overlapping boxes"""
    if len(boxes) == 0:
        return boxes, scores, classes
    
    # Convert to tensors if needed
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
    # Apply NMS
    keep_indices = nms(boxes, scores, iou_threshold)
    
    return boxes[keep_indices], scores[keep_indices], classes[keep_indices]


def temporal_filter(current_detections, history):
    """
    Filter detections using temporal smoothing.
    Only show detections that appear consistently across frames.
    """
    if len(history) < MIN_DETECTION_FRAMES:
        return current_detections
    
    # Count class occurrences in recent frames
    class_counts = {}
    for frame_dets in history:
        seen_classes = set()
        for det in frame_dets:
            cls = det['class']
            if cls not in seen_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1
                seen_classes.add(cls)
    
    # Filter current detections - keep only if class appeared in enough frames
    stable_detections = []
    for det in current_detections:
        cls = det['class']
        if class_counts.get(cls, 0) >= MIN_DETECTION_FRAMES:
            stable_detections.append(det)
    
    return stable_detections


def smooth_bbox(current_bbox, history, class_name):
    """Smooth bounding box coordinates using exponential moving average"""
    if len(history) == 0:
        return current_bbox
    
    # Find matching class detections in history
    past_bboxes = []
    for frame_dets in history:
        for det in frame_dets:
            if det['class'] == class_name:
                past_bboxes.append(det['bbox'])
                break
    
    if len(past_bboxes) == 0:
        return current_bbox
    
    # Exponential moving average
    alpha = 0.6  # Weight for current frame (higher = less smoothing)
    smoothed = current_bbox.copy()
    avg_past = np.mean(past_bboxes, axis=0)
    
    for i in range(4):
        smoothed[i] = alpha * current_bbox[i] + (1 - alpha) * avg_past[i]
    
    return smoothed


while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame from camera")
        break
    
    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Get actual frame dimensions
    frame_h, frame_w = frame.shape[:2]
        
    # Time the inference
    inference_start = time.time()
    transformed = transforms(image=frame)
    input_tensor = torch.unsqueeze(transformed['image'], dim=0).to(device)
    
    with torch.no_grad():
        result = model(input_tensor)
    inference_time = (time.time() - inference_start) * 1000  # Convert to ms

    # Get probabilities (exclude "no object" class which is the last one)
    all_probs = result['pred_logits'].softmax(-1)
    probabilities = all_probs[:, :, :-1]  # Object classes only
    no_obj_probs = all_probs[:, :, -1]    # "No object" probability
    
    max_probs, max_classes = probabilities.max(-1)
    
    # Keep detections where:
    # 1. Confidence > threshold
    # 2. Object probability > "no object" probability (reduces false positives)
    keep_mask = (max_probs > CONFIDENCE_THRESHOLD) & (max_probs > no_obj_probs * 1.2)

    batch_indices, query_indices = torch.where(keep_mask) 

    # Use actual frame dimensions instead of hardcoded values
    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (frame_w, frame_h))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    # Apply Non-Maximum Suppression to reduce overlapping boxes
    if len(bboxes) > 0:
        bboxes, probas, classes = apply_nms(bboxes, probas, classes, NMS_THRESHOLD)

    # Prepare detection results
    raw_detections = []
    for bclass, bprob, bbox in zip(classes, probas, bboxes): 
        bclass_idx = bclass.detach().cpu().numpy()
        bprob_val = bprob.detach().cpu().numpy() 
        x1, y1, x2, y2 = bbox.detach().cpu().numpy()
        
        raw_detections.append({
            'class': CLASSES[bclass_idx],
            'confidence': float(bprob_val),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
    
    # Add to history for temporal smoothing
    detection_history.append(raw_detections)
    
    # Apply temporal filtering for stability
    stable_detections = temporal_filter(raw_detections, detection_history)
    
    # Draw stable detections with smoothed boxes
    for det in stable_detections:
        class_name = det['class']
        confidence = det['confidence']
        bbox = smooth_bbox(det['bbox'], detection_history, class_name)
        
        x1, y1, x2, y2 = bbox
        class_idx = CLASSES.index(class_name)
        
        # Draw bounding box with rounded corners effect (thinner line)
        color = COLORS[class_idx]
        thickness = 3
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Draw label background (smaller, cleaner)
        label_text = f"{class_name}: {confidence:.1%}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        label_y = int(y1) - 10 if int(y1) - 10 > text_h else int(y1) + text_h + 10
        frame = cv2.rectangle(frame, (int(x1), label_y - text_h - 5), (int(x1) + text_w + 10, label_y + 5), color, -1)
        frame = cv2.putText(frame, label_text, (int(x1) + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Calculate FPS
    frame_count += 1
    if frame_count % 30 == 0:  # Log every 30 frames
        elapsed_time = time.time() - fps_start_time
        fps = 30 / elapsed_time
        
        # Log detection results and performance
        if stable_detections:
            detection_handler.log_detections(stable_detections, frame_id=frame_count)
        detection_handler.log_inference_time(inference_time, fps)
        
        # Reset FPS counter
        fps_start_time = time.time()

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {1000/inference_time:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        logger.realtime("Stopping real-time detection...")
        break

cap.release() 
cv2.destroyAllWindows() 
