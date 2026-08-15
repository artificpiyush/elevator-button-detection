# Elevator Button Detection and Recognition Pipeline

This project implements an end-to-end computer vision pipeline that detects elevator buttons from an image, extracts text/numbers on each button using Optical Character Recognition (OCR), and identifies the specific button corresponding to a user-requested target floor using high-precision string and visual confusion matching.

## Features
- **Object Detection**: YOLOv8n fine-tuned on elevator buttons.
- **Text Recognition**: EasyOCR (CRNN) fine-tuned on elevator button crops (with CRAFT detection bypassed for tight crops).
- **Precision-First Floor Selection (`FloorSelector2`)**: Combines exact string matching, elevator token synonym/ligature resolution, and visually-weighted Levenshtein distance with strict thresholds.
- **Robot Safety Principle**: If a target floor cannot be matched with high visual confidence, the system safely returns `Target floor button not found` rather than risk pressing the wrong floor.

---

## 1. Dataset Used
- **Detection Dataset**: SMU Elevator Button dataset, containing varied images of elevator panels under different lighting conditions.
- **OCR Dataset**: Cropped button images mapped to their text labels, augmented for variance in glare, noise, and metallic reflection.

## 2. Dataset Preparation
- The YOLO dataset was mapped to a unified `button` class (1-class schema) instead of differentiating between screens and buttons.
- The OCR dataset was formatted into a `lmdb`-style mapping (`gt.txt`) for PyTorch-based text recognition training.

## 3. Training Configuration & Hyperparameters
### YOLOv8
- **Base Model**: YOLOv8 nano (`yolov8n.pt`)
- **Epochs**: 50
- **Image Size**: 640
- **Optimizer**: Default (AdamW/SGD auto-selected by Ultralytics)
- **Batch Size**: 16

### EasyOCR (CRNN)
- **Architecture**: ResNet feature extraction + BiLSTM sequence modeling + CTC Loss.
- **Epochs**: 30
- **Optimizer**: Adam (lr = 1e-4)
- **Vocabulary**: Alphanumeric (36 characters: a-z, 0-9) via `charset_36_EN.txt`

## 4. Pipeline Architecture
The pipeline consists of three modular components:
1. **Detector (`src/detector.py`)**: Runs fine-tuned YOLOv8 to extract button bounding boxes and centers.
2. **OCR Extractor (`src/ocr.py`)**: Takes cropped bounding boxes, bypasses standard CRAFT text-detection, and feeds the image directly into the fine-tuned CRNN recognizer.
3. **Floor Selector (`src/floor_selector_2.py`)**: High-precision matcher that pairs extracted text with the requested target floor without false-positive collisions.

## 5. Floor Selection Architecture (`FloorSelector2`)
Elevator panels frequently suffer from metallic glare, low contrast, character confusions (e.g. `l` $\leftrightarrow$ `1`, `O` $\leftrightarrow$ `0`), and ligature splits (e.g. `1l` $\leftrightarrow$ `M`). `FloorSelector2` tackles this using 2 verified, precision-first techniques:

1. **Elevator Token Synonym & Ligature Resolution**: Resolves common OCR artifacts and segmentations (e.g., `'112'` $\rightarrow$ `'12'`, `'1l'` $\rightarrow$ `'M'`, `'1M'` $\rightarrow$ `'9M'`).
2. **Visually-Weighted Levenshtein Edit Distance with Strict Thresholds**: Substitutions between visually similar characters (`0`/`O`/`D`/`Q`, `1`/`l`/`I`, `8`/`B`, `5`/`S`, `2`/`Z`, `6`/`G`, `4`/`A`) incur a penalty of only 0.2, but distinct floor digits (e.g. `3` vs `8`, `7` vs `1`) are strictly isolated to avoid hazardous robot button presses.
3. **Safe Fallback**: If no button meets the strict visual distance threshold, the system returns `Target floor button not found`.

## 6. Evaluation Results

### Detection (YOLOv8 1-Class)
- **Precision**: 0.94
- **Recall**: 0.91
- **mAP@50**: 0.95
- **mAP@50:95**: 0.72
- **Inference latency**: ~55-70ms

### OCR Recognizer (Fine-Tuned CRNN)
- **Evaluated Buttons**: 250
- **Correctly Recognized**: 228
- **Accuracy**: 91.2%

### End-to-End Floor Matching (37 Test Cases across 8 Test Panels)

| Component | Correct / Total Cases | Accuracy | Precision (False Positive Rate) |
| :--- | :---: | :---: | :---: |
| **Baseline `FloorSelector` (Exact Match)** | 20 / 37 | **54.1%** | 100% (0% false positives) |
| **Enhanced `FloorSelector2` (Visual & Ligature Match)** | **25 / 37** | **67.6%** | **100% (0% false positives)** |

*All results, annotated visual outputs, and individual JSON predictions are saved in `examples2/` along with `examples2/evaluation_report.csv`.*

## 7. Inference Instructions

### Run End-to-End Floor Detection
```bash
python detect_floor.py --image examples2/003_png.rf.85f1564ada6fbb04a7ab4ca0f14648e9.jpg --target 4
```

### Module Usage Example
```python
from detect_floor import detect_floor

result = detect_floor(
    image_path="examples2/003_png.rf.85f1564ada6fbb04a7ab4ca0f14648e9.jpg",
    target_floor="4"
)
print(result)
```
