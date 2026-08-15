import os
import cv2
import easyocr
import torch

class OCRExtractor:
    """
    OCR recognition engine using EasyOCR with fine-tuned recognizer weights.
    """
    def __init__(self, finetuned_model_path, lang=['en']):
        self.reader = easyocr.Reader(lang)
        
        if os.path.exists(finetuned_model_path):
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(finetuned_model_path, map_location=device)
            
            # Handle DataParallel/DistributedDataParallel state dict prefixes if present
            if hasattr(self.reader.recognizer, 'module'):
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k if k.startswith('module.') else f'module.{k}'
                    new_state_dict[new_key] = v
                state_dict = new_state_dict
                
            self.reader.recognizer.load_state_dict(state_dict)
            self.reader.recognizer.eval()
            
    def extract_text(self, image_crop):
        """
        Extracts character text directly from a cropped button image.
        Bypasses CRAFT text detector since bounding box is already cropped.
        """
        gray_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray_crop.shape[:2]
        
        # Pass bounding box covering the entire crop directly to recognizer
        horizontal_list = [[0, w, 0, h]]
        
        ocr_result = self.reader.recognize(gray_crop, horizontal_list=horizontal_list, free_list=[])
        
        if ocr_result:
            ocr_result.sort(key=lambda x: x[2], reverse=True)
            return ocr_result[0][1].strip()
            
        return ""
