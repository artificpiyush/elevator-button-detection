import argparse
import json
import os
import cv2
import sys

# Add src to module search path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from detector import ButtonDetector
from ocr import OCRExtractor
from floor_selector_2 import FloorSelector2

def detect_floor(image_path, target_floor, yolo_model_path="models/best_yolo.pt"):
    """
    Main floor detection pipeline: runs object detection, OCR, and target selection.
    """
    if not os.path.exists(yolo_model_path) and os.path.exists("yolov8n.pt"):
        yolo_model_path = "yolov8n.pt"
        
    detector = ButtonDetector(yolo_model_path)
    
    ocr_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_easyocr_finetuned.pt')
    ocr = OCRExtractor(ocr_model_path)
    
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not read image at {image_path}"}
        
    detected_boxes, speed = detector.detect(image)
    
    buttons = []
    pad = 2
    h, w = image.shape[:2]

    for btn in detected_boxes:
        x_min, y_min, x_max, y_max = btn['bbox']
        
        crop_y_min = max(0, y_min - pad)
        crop_y_max = min(h, y_max + pad)
        crop_x_min = max(0, x_min - pad)
        crop_x_max = min(w, x_max + pad)
        
        crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        btn['text'] = ocr.extract_text(crop)
        buttons.append(btn)
        
    best_match = FloorSelector2.find_target_button(buttons, target_floor)
    
    if best_match:
        return {
            "target_floor": target_floor,
            "button": best_match,
            "speed": speed
        }
    else:
        return {
            "target_floor": target_floor,
            "error": "Target floor button not found",
            "all_detected_buttons": buttons,
            "speed": speed
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elevator Floor Button Locator")
    parser.add_argument("--image", required=True, help="Path to input elevator image")
    parser.add_argument("--target", type=str, required=True, help="Target floor identifier (e.g. '3', 'M', '12')")
    parser.add_argument("--model", default="models/best_yolo.pt", help="Path to YOLO model checkpoint")
    
    args = parser.parse_args()
    
    result = detect_floor(args.image, args.target, args.model)
    print(json.dumps(result, indent=2))
