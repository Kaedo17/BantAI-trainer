#Import YOLO
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("Initializing YOLOv10-Nano for BantAI...")
    
    # Optional: Print GPU info for verification
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    #pre-trained Nano model
    model = YOLO("yolov10n.pt") 

    print("Starting training phase. This may take a few hours...")
    results = model.train(
        data="archive/Open-Flame-Hazards-YOLO/data.yaml", 
        epochs=150,
        imgsz=640,
        augment=True,
        workers=2,           # Your current setting - good for Windows
        batch=16,            # Your current setting - conservative
        device=0,
        project="BantAI",
        name="Model_v1",
        patience=50,
        save=True,           
        exist_ok=True,       
        optimizer='auto',    
        verbose=True,        
        amp=True,            # Mixed precision - good for RTX 5050
        cache=False,         # Your current setting - avoids RAM issues
        close_mosaic=10,
        # Add these for better stability:
        mosaic=1.0,          # Keep mosaic augmentation (helps with small objects)
        mixup=0.0,           # Disable mixup initially (can be memory intensive)
        copy_paste=0.0       # Disable copy-paste initially
    )

    print("Training complete! Exporting to Android ExecuTorch format...")

    # Export the best performing model
    best_path = results.save_dir / "weights" / "best.pt"
    print(f"Loading best model from: {best_path}")
    best_model = YOLO(str(best_path))
    best_model.export(format="executorch", optimize=True)

    print(f"Export successful! Your .pte file is ready to be downloaded.")
    print(f"Model saved to: {results.save_dir}")