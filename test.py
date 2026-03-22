#!/usr/bin/env python3
"""
Model testing with adjustable confidence threshold
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from config import get_gpu_info, print_banner
from utils import get_folder_path

def main():
    print_banner("Model Testing")
    
    # Get model path
    print("\n📁 Select your trained model")
    model_path = get_folder_path("Enter path to model file (.pt)")
    
    if not model_path.suffix == '.pt':
        print("❌ Please select a .pt model file")
        return
    
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
    print("   1. Test on image file")
    print("   2. Test on folder of images")
    print("   3. Test on video file")
    print("   4. Test on webcam")
    print("   5. Test on URL")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        # Single image
        image_path = get_folder_path("Enter image file path")
        if image_path.is_file():
            results = model.predict(
                str(image_path),
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                show=True
            )
            print(f"\n✅ Detection complete! Results saved to runs/detect/")
        else:
            print("❌ Invalid file path")
            
    elif choice == '2':
        # Folder of images
        folder_path = get_folder_path("Enter folder path with images")
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
        video_path = get_folder_path("Enter video file path")
        if video_path.is_file():
            results = model.predict(
                str(video_path),
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                show=True
            )
            print(f"\n✅ Video processed!")
        else:
            print("❌ Invalid file path")
            
    elif choice == '4':
        # Webcam
        cam_id = input("Enter camera ID (default: 0): ").strip() or "0"
        try:
            cam_id = int(cam_id)
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
        # URL
        url = input("Enter image URL: ").strip()
        if url:
            results = model.predict(
                url,
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                show=True
            )
            print(f"\n✅ Detection complete!")
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