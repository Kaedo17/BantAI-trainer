#!/usr/bin/env python3
"""
Model testing with adjustable confidence threshold
"""

import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from config import get_gpu_info, print_banner

def get_file_path(prompt, extension=".pt"):
    """Get file path from user with validation"""
    while True:
        path = input(f"{prompt}: ").strip().strip('"')
        if not path:
            print("❌ Path cannot be empty. Please try again.")
            continue
        path = Path(path)
        if not path.exists():
            print(f"❌ Path '{path}' does not exist. Please try again.")
            continue
        if not path.is_file():
            print(f"❌ '{path}' is not a file. Please try again.")
            continue
        if path.suffix != extension:
            print(f"❌ Please select a {extension} file. Got: {path.suffix}")
            continue
        return path

def get_folder_path(prompt):
    """Get folder path from user with validation"""
    while True:
        path = input(f"{prompt}: ").strip().strip('"')
        if not path:
            print("❌ Path cannot be empty. Please try again.")
            continue
        path = Path(path)
        if not path.exists():
            print(f"❌ Path '{path}' does not exist. Please try again.")
            continue
        if not path.is_dir():
            print(f"❌ '{path}' is not a directory. Please try again.")
            continue
        return path

def main():
    print_banner("Model Testing")
    
    # Get model path - use file picker instead of folder picker
    print("\n📁 Select your trained model (.pt file)")
    print("   Example: runs/detect/Open_Flame_Model/Model_1/weights/best.pt")
    model_path = get_file_path("Enter path to model file (.pt)", extension=".pt")
    
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
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        # Single image
        print("\n📸 Select an image file")
        image_path = get_file_path("Enter image file path", extension="")
        if image_path.is_file():
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
        folder_path = get_folder_path("Enter folder path with images")
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
        video_path = get_file_path("Enter video file path", extension="")
        if video_path.is_file():
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