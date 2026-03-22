#!/usr/bin/env python3
"""
YOLO Training Pipeline - Main Controller
A unified interface for augmenting, splitting, training, and testing YOLO models
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import print_banner, get_gpu_info
from utils import get_folder_path, get_output_path


class YOLOPipeline:
    """Main controller for YOLO training pipeline"""
    
    def __init__(self):
        self.scripts_dir = Path(__file__).parent
        self.available_scripts = {
            'augment': 'augment.py',
            'split': 'split.py',
            'train': 'train.py',
            'test': 'test.py'
        }
        
    def print_menu(self):
        """Print the main menu"""
        print_banner("YOLO Training Pipeline")
        
        print("\n📋 Available Options:")
        print("   ┌─────────────────────────────────────────────────┐")
        print("   │ 1.  Augment Images                              │")
        print("   │ 2.  Split Dataset (80% Train, 10% Val, 10% Test)│")
        print("   │ 3.  Train Model                                 │")
        print("   │ 4.  Test Model                                  │")
        print("   │ 5.  Complete Pipeline (Augment → Split → Train) │")
        print("   │ 6.  Quick Train (Skip Augment/Split)            │")
        print("   │ 7.  View GPU Info                               │")
        print("   │ 8.  Exit                                        │")
        print("   └─────────────────────────────────────────────────┘")
        
    def run_script(self, script_name, *args):
        """Run a Python script with arguments"""
        script_path = self.scripts_dir / self.available_scripts[script_name]
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        print(f"\n🚀 Running {script_name}...")
        print("="*60)
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)] + list(args),
                cwd=self.scripts_dir,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")
            return False
    
    def augment(self):
        """Run augmentation"""
        return self.run_script('augment')
    
    def split(self):
        """Run dataset splitting"""
        return self.run_script('split')
    
    def train(self):
        """Run training"""
        return self.run_script('train')
    
    def test(self):
        """Run testing"""
        return self.run_script('test')
    
    def complete_pipeline(self):
        """Run complete pipeline: augment → split → train"""
        print_banner("Complete Pipeline")
        print("\nThis will run: Augment → Split → Train")
        print("Make sure you have your dataset ready in the correct format.")
        print("\n⚠️  Note: If your data is already split, consider using 'Quick Train' instead.")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled")
            return False
        
        steps = [
            ('augment', "Step 1: Augmenting images..."),
            ('split', "Step 2: Splitting dataset..."),
            ('train', "Step 3: Training model...")
        ]
        
        for step_name, message in steps:
            print(f"\n{'='*60}")
            print(f"  {message}")
            print('='*60)
            
            success = self.run_script(step_name)
            if not success:
                print(f"❌ Pipeline failed at {step_name}")
                return False
            
            # Ask to continue after each step
            if step_name != steps[-1][0]:
                cont = input("\nContinue to next step? (y/n): ").strip().lower()
                if cont != 'y':
                    print("❌ Pipeline stopped by user")
                    return False
        
        print_banner("Pipeline Complete!")
        print("\n✅ All steps completed successfully!")
        print("   Your model is ready for testing.")
        return True
    
    def quick_train(self):
        """Quick train using existing dataset (skip augment/split)"""
        print_banner("Quick Train")
        print("\nThis will train a model using your existing dataset.")
        print("Make sure your dataset is already organized with:")
        print("   - images/train, images/val, images/test")
        print("   - labels/train, labels/val, labels/test")
        print("   - data.yaml in the root folder")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled")
            return False
        
        return self.run_script('train')
    
    def show_gpu_info(self):
        """Display GPU information"""
        print_banner("GPU Information")
        
        gpu = get_gpu_info()
        if gpu:
            print(f"\n🎮 GPU: {gpu['name']}")
            print(f"   VRAM: {gpu['vram_gb']:.2f} GB")
            print(f"   CUDA Available: Yes")
            
            # Try to get more info
            try:
                import torch
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   PyTorch Version: {torch.__version__}")
                print(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}")
            except:
                pass
        else:
            print("\n⚠️  No GPU detected. Training will be slow on CPU.")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main run loop"""
        while True:
            self.print_menu()
            
            choice = input("\n👉 Select an option (1-8): ").strip()
            
            if choice == '1':
                self.augment()
            elif choice == '2':
                self.split()
            elif choice == '3':
                self.train()
            elif choice == '4':
                self.test()
            elif choice == '5':
                self.complete_pipeline()
            elif choice == '6':
                self.quick_train()
            elif choice == '7':
                self.show_gpu_info()
            elif choice == '8':
                print_banner("Goodbye!")
                print("\n👋 Thanks for using YOLO Training Pipeline!")
                break
            else:
                print("❌ Invalid option. Please enter 1-8.")
            
            input("\nPress Enter to continue...")


def main():
    """Main entry point with command-line argument support"""
    parser = argparse.ArgumentParser(description='YOLO Training Pipeline')
    parser.add_argument('--mode', type=str, choices=['augment', 'split', 'train', 'test', 'all', 'quick'],
                        help='Run specific mode directly without menu')
    parser.add_argument('--dataset', type=str, help='Dataset path (for train mode)')
    parser.add_argument('--model', type=str, default='yolov26s.pt', help='Model name')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for test')
    
    args = parser.parse_args()
    
    pipeline = YOLOPipeline()
    
    # If mode is specified, run directly
    if args.mode:
        print_banner(f"Running in {args.mode} mode")
        
        if args.mode == 'augment':
            pipeline.augment()
        elif args.mode == 'split':
            pipeline.split()
        elif args.mode == 'train':
            pipeline.train()
        elif args.mode == 'test':
            pipeline.test()
        elif args.mode == 'all':
            pipeline.complete_pipeline()
        elif args.mode == 'quick':
            pipeline.quick_train()
        return
    
    # Otherwise show interactive menu
    pipeline.run()


if __name__ == '__main__':
    main()