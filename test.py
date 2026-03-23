#!/usr/bin/env python3
"""
Model testing with adjustable confidence threshold
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from config import get_gpu_info, print_banner
from utils import get_file_path, get_folder_path, ROOT_DIR

# Define trained models base
TRAINED_BASE = ROOT_DIR / "Trained"

def browse_trained_models():
    """Browse and select from trained models"""
    if not TRAINED_BASE.exists():
        print(f"\n❌ No trained models found in {TRAINED_BASE}")
        return None
    
    models = []
    for model_dir in TRAINED_BASE.iterdir():
        if model_dir.is_dir():
            # Check for best.pt in various locations
            for weight_path in [
                model_dir / "weights" / "best.pt",
                model_dir / "best.pt",
                model_dir / "weights" / "weights" / "best.pt"
            ]:
                if weight_path.exists():
                    models.append((model_dir.name, weight_path))
                    break
    
    if not models:
        print(f"\n❌ No model files found in {TRAINED_BASE}")
        return None
    
    print(f"\n📁 Trained Models in {TRAINED_BASE}:")
    for i, (name, path) in enumerate(models):
        print(f"   {i+1}. {name} -> {path}")
    
    try:
        choice = input("\nSelect model number (or 0 to enter custom path): ").strip()
        if choice == '0':
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx][1]
    except:
        pass
    
    return None

def main():
    print_banner("Model Testing")
    
    # Get model path - try to browse first
    print("\n📁 Select your trained model (.pt file)")
    print("   Trained models are stored in: Trained/")
    
    model_path = browse_trained_models()
    
    if model_path is None:
        # Manual entry
        model_path = get_file_path("Enter path to model file (.pt)", extension=".pt", allow_cancel=True)
        if model_path == 'BACK':
            return
        if model_path == 'CANCEL':
            print("❌ Cancelled by user")
            return
    
    print(f"\n✅ Using model: {model_path}")
    
    # Get confidence threshold
    try:
        conf_threshold = float(input("Enter confidence threshold (0-1, default: 0.5): ").strip() or "0.5")
        conf_threshold = max(0.0, min(1.0, conf_threshold))
    except ValueError:
        conf_threshold = 0.5
    
    # Get IOU threshold
    try:
        iou_threshold = float(input("Enter IOU threshold for NMS (0-1, default: 0.45): ").strip() or "0.45")
        iou_threshold = max(0.0, min(1.0, iou_threshold))
    except ValueError:
        iou_threshold = 0.45
    
    print(f"\n⚙️ Detection settings:")
    print(f"   Confidence threshold: {conf_threshold:.2f}")
    print(f"   IOU threshold: {iou_threshold:.2f}")
    
    # Load model
    print(f"\n📥 Loading model: {model_path}")
    try:
        model = YOLO(str(model_path))
        
        # Show GPU info
        gpu = get_gpu_info()
        if gpu:
            print(f"   Running on: {gpu['name']}")
        else:
            print("   Running on: CPU")
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Testing options
    print("\n📋 Testing options:")
    print("   1. Test on single image file")
    print("   2. Test on folder of images")
    print("   3. Test on video file")
    print("   4. Test on webcam")
    print("   5. Test on image URL")
    print("   0. Back to main menu")
    
    choice = input("\nSelect option (0-5): ").strip()
    
    if choice == '0':
        return
    
    elif choice == '1':
        # Single image
        print("\n📸 Select an image file")
        image_path = get_file_path("Enter image file path", extension="", allow_cancel=True)
        if image_path == 'BACK':
            return
        if image_path == 'CANCEL':
            print("❌ Cancelled by user")
            return
        if image_path and image_path.is_file():
            print(f"\n🔍 Running inference on: {image_path.name}")
            results = model.predict(
                str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                show=True
            )
            # Display results
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"\n✅ Detected {len(boxes)} objects:")
                    for i, box in enumerate(boxes):
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        print(f"   {i+1}. Class: {cls}, Confidence: {conf:.2%}")
                else:
                    print("\n⚠️ No objects detected with current confidence threshold")
            print(f"\n📁 Results saved to runs/detect/")
        else:
            print("❌ Invalid file path")
            
    elif choice == '2':
        # Folder of images
        print("\n📁 Select a folder containing images")
        folder_path = get_folder_path("Enter folder path with images", allow_cancel=True)
        if folder_path == 'BACK':
            return
        if folder_path == 'CANCEL':
            print("❌ Cancelled by user")
            return
        if folder_path:
            print(f"\n🔍 Running inference on all images in: {folder_path}")
            results = model.predict(
                str(folder_path),
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                show=False
            )
            print(f"\n✅ Processed {len(results)} images")
            print(f"   Results saved to runs/detect/predict/")
        
    elif choice == '3':
        # Video file
        print("\n🎥 Select a video file")
        video_path = get_file_path("Enter video file path", extension="", allow_cancel=True)
        if video_path == 'BACK':
            return
        if video_path == 'CANCEL':
            print("❌ Cancelled by user")
            return
        if video_path and video_path.is_file():
            print(f"\n🔍 Running inference on video: {video_path.name}")
            results = model.predict(
                str(video_path),
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                show=True
            )
            print(f"\n✅ Video processed! Results saved to runs/detect/")
        else:
            print("❌ Invalid file path")
            
    elif choice == '4':
        # Webcam
        cam_id = input("Enter camera ID (default: 0): ").strip() or "0"
        try:
            cam_id = int(cam_id)
            print(f"\n🎥 Opening webcam {cam_id}... Press 'q' to quit")
            results = model.predict(
                source=cam_id,
                conf=conf_threshold,
                iou=iou_threshold,
                show=True,
                stream=True
            )
        except Exception as e:
            print(f"❌ Webcam error: {e}")
            
    elif choice == '5':
        # Image URL
        url = input("Enter image URL: ").strip()
        if url:
            print(f"\n🔍 Running inference on URL")
            results = model.predict(
                url,
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                show=True
            )
            print(f"\n✅ Detection complete! Results saved to runs/detect/")
        else:
            print("❌ No URL provided")
    
    else:
        print("❌ Invalid option")

def quick_test(model_path, image_path=None, conf=0.5, iou=0.45):
    """Quick test function for programmatic use"""
    model = YOLO(model_path)
    if image_path:
        results = model.predict(image_path, conf=conf, iou=iou, show=True)
        return results
    return None

if __name__ == '__main__':
    main()