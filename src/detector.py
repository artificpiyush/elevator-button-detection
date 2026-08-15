import os
from ultralytics import YOLO

class ButtonDetector:
    """
    Button detector wrapping YOLOv8 to locate elevator buttons.
    """
    def __init__(self, model_path):
        # Resolve weight directory or direct checkpoint path
        if os.path.isdir(model_path):
            weights_path = os.path.join(model_path, 'weights', 'best.pt')
            if os.path.exists(weights_path):
                model_path = weights_path
            else:
                fallback_path = os.path.join(model_path, 'best.pt')
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                    
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Runs object detection on the input image.
        Returns a list of detected button dictionaries (bbox, center, confidence) and inference latency stats.
        """
        results = self.model(image)[0]
        detected_buttons = []
        
        for box in results.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            
            detected_buttons.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "center": [cx, cy],
                "confidence": conf
            })
            
        return detected_buttons, results.speed
