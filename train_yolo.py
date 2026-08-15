import os
from dotenv import load_dotenv
import wandb
from ultralytics import YOLO

def main():
    """
    Fine-tunes YOLOv8n on the elevator button detection dataset.
    """
    load_dotenv()
    
    # Optional experiment tracking with Weights & Biases
    if os.getenv("WANDB_API_KEY"):
        wandb.login()
        wandb.init(
            project="elevator-button-detection", 
            name="yolov8n-smu-binary"
        )
    
    # Load pretrained YOLOv8n backbone
    model = YOLO("yolov8n.pt")
    
    data_yaml_path = os.path.abspath("data/smu_dataset_binary/data.yaml")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Dataset config not found at: {data_yaml_path}")
    
    # Start training
    results = model.train(
        data=data_yaml_path,
        epochs=10,
        imgsz=640,
        batch=16,
        device="mps",
        project="models",
        name="yolo_button_detector",
        optimizer="auto",
        patience=10,
        save=True,
        plots=True,
        val=True
    )
    
    if os.getenv("WANDB_API_KEY"):
        wandb.finish()
        
    print("Training complete. Checkpoints saved to models/yolo_button_detector/weights/best.pt")

if __name__ == "__main__":
    main()
