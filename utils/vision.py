import cv2
import numpy as np
from ultralytics import YOLO

from utils.labels import to_pt

class ObstacleDetector:
    # Curated whitelist of COCO classes relevant for assistive navigation.
    # YOLOv8n detects all 80 COCO classes; we filter to only report useful ones.
    DEFAULT_ASSISTIVE_CLASSES = {
        # People
        'person',
        # Vehicles (outdoor navigation)
        'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        # Animals
        'dog', 'cat', 'bird',
        # Street furniture & obstacles
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        # Everyday indoor items
        'chair', 'couch', 'dining table', 'bed',
        'toilet', 'sink',
        'refrigerator', 'oven', 'microwave',
        'tv', 'laptop', 'cell phone',
        # Portable objects the user may bump into
        'bottle', 'cup', 'backpack', 'umbrella', 'handbag', 'suitcase',
        'book', 'vase', 'potted plant',
    }

    def __init__(self, model_size='yolov8n.pt', allowed_classes=None):
        """ Initialize the YOLOv8 model for object detection.
        
        Args:
            model_size: YOLO model weight file to load.
            allowed_classes: Optional set of class names to detect.
                             Defaults to DEFAULT_ASSISTIVE_CLASSES.
        """
        # 'n' is nano, meaning it's the fastest and smallest model. Perfect for real-time.
        print(f"Loading YOLO model {model_size}...")
        self.model = YOLO(model_size)
        self.allowed_classes = allowed_classes or self.DEFAULT_ASSISTIVE_CLASSES
        print(f"YOLO model loaded. Tracking {len(self.allowed_classes)} object categories.")

    def process_frame(self, color_img, depth_img, confidence_threshold=0.5):

        # 1. Run inference on the color image
        results = self.model(color_img, verbose=False)
        result = results[0] # Get the first (and only) image result
        
        annotated_img = color_img.copy()
        threats = []

        # 2. Extract bounding boxes and check depth
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf < confidence_threshold:
                continue
            
            # Get class name (e.g., 'person', 'chair', 'car')
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id]

            # Skip classes not in the assistive whitelist
            if class_name not in self.allowed_classes:
                continue
            
            # Get bounding box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # 3. Calculate Distance
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            height, width = depth_img.shape
            patch = 5  # half-size of sampling patch in pixels
            y1p = max(0, center_y - patch)
            y2p = min(height, center_y + patch + 1)
            x1p = max(0, center_x - patch)
            x2p = min(width, center_x + patch + 1)
            region = depth_img[y1p:y2p, x1p:x2p]
            valid = region[region > 0]
            distance_mm = int(np.median(valid)) if valid.size > 0 else 0
            # 0 means no valid readings in the patch (object too close or too far)
            
            threats.append({
                'label': class_name,
                'label_pt': to_pt(class_name),
                'distance_mm': distance_mm,
                'bbox': (x1, y1, x2, y2)
            })
            
            # 4. Draw bounding box and info on the image for debugging
            dist_str = f"{distance_mm/1000:.1f}m" if distance_mm > 0 else "N/A"
            label_text = f"{class_name} {dist_str}"
            
            # Red box if within 1 meter, otherwise green
            color = (0, 0, 255) if (0 < distance_mm < 1000) else (0, 255, 0)
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_img, threats

# Simple test if run directly
if __name__ == '__main__':
    detector = ObstacleDetector()
    # Create fake blank images to test if it runs without crashing
    c_img = np.zeros((480, 640, 3), dtype=np.uint8)
    d_img = np.zeros((480, 640), dtype=np.uint16)
    c, t = detector.process_frame(c_img, d_img)
    print(f"Found objects: {t}")
