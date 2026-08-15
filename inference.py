import argparse
import json
import os
import cv2
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from detector import ButtonDetector
from ocr import OCRExtractor
from floor_selector_2 import FloorSelector2

class ElevatorButtonDetector:
    def __init__(self, yolo_model_path="models/best_yolo.pt"):
        if not os.path.exists(yolo_model_path) and os.path.exists("yolov8n.pt"):
            yolo_model_path = "yolov8n.pt"
            
        self.detector = ButtonDetector(yolo_model_path)
        
        ocr_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_easyocr_finetuned.pt')
        self.ocr = OCRExtractor(ocr_model_path)

    def detect_and_recognize(self, image_path, target_floor=None):
        """
        Run detection and OCR on the image.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # 1. Detect buttons
        detected_boxes, speed = self.detector.detect(image)
        
        buttons = []
        pad = 2
        h, w = image.shape[:2]

        for btn in detected_boxes:
            x_min, y_min, x_max, y_max = btn['bbox']
            
            # Crop image for OCR
            crop_y_min = max(0, y_min - pad)
            crop_y_max = min(h, y_max + pad)
            crop_x_min = max(0, x_min - pad)
            crop_x_max = min(w, x_max + pad)
            
            crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            # 2. Extract text with OCR
            btn['text'] = self.ocr.extract_text(crop)
            buttons.append(btn)
            
        # 3. Identify target floor using FloorSelector2
        if target_floor is not None:
            best_match = FloorSelector2.find_target_button(buttons, target_floor)
            if best_match:
                return {
                    "target_floor": target_floor,
                    "button": best_match
                }
            else:
                return {
                    "target_floor": target_floor,
                    "error": "Target floor button not found",
                    "all_detected_buttons": buttons
                }
                
        return buttons

def main():
    parser = argparse.ArgumentParser(description="Elevator Button Detection and Floor Recognition")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/best_yolo.pt", help="Path to YOLO model checkpoint")
    parser.add_argument("--target", type=str, required=True, help="Target floor number or text")
    
    args = parser.parse_args()
    
    try:
        pipeline = ElevatorButtonDetector(yolo_model_path=args.model)
        result = pipeline.detect_and_recognize(args.image, target_floor=args.target)
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))

if __name__ == "__main__":
    main()
