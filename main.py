#!/usr/bin/env python3
"""
Car Damage Analysis - Main Entry Point
=====================================

This script provides a simple way to run the complete car damage analysis pipeline
from data preprocessing to model training and evaluation.

Usage:
    python main.py --help
    python main.py --mode preprocessing
    python main.py --mode training
    python main.py --mode evaluation
    python main.py --mode demo
    python main.py --mode all

Author: AI Engineer
Date: 2024
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickStartPipeline:
    """Quick start pipeline for car damage analysis."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_root = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.scripts_dir = self.project_root / "scripts"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        (self.models_dir / "detection").mkdir(exist_ok=True)
        (self.models_dir / "segmentation").mkdir(exist_ok=True)
        (self.models_dir / "severity").mkdir(exist_ok=True)
    
    def check_prerequisites(self):
        """Check if all prerequisites are installed."""
        logger.info("Checking prerequisites...")
        
        try:
            import torch
            import ultralytics
            import cv2
            import pandas as pd
            import numpy as np
            logger.info(f"✓ PyTorch {torch.__version__}")
            logger.info(f"✓ Ultralytics {ultralytics.__version__}")
            logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Please run: pip install -r requirements.txt")
            return False
        
        # Check data directory
        coco_dir = self.data_root / "CarDD_COCO"
        if not coco_dir.exists():
            logger.error(f"Data directory not found: {coco_dir}")
            logger.error("Please place the CarDD COCO dataset in data/CarDD_COCO/")
            return False
        
        logger.info("✓ All prerequisites met")
        return True
    
    def run_preprocessing(self):
        """Run data preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        
        cmd = [
            sys.executable, 
            str(self.scripts_dir / "coco_data_preprocessing.py"),
            "--data_root", str(self.data_root / "CarDD_COCO"),
            "--output_root", str(self.data_root / "processed")
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✓ Data preprocessing completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Data preprocessing failed: {e}")
            logger.error(f"Output: {e.stdout}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def run_detection_training(self):
        """Run detection model training."""
        logger.info("Starting detection model training...")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "train_yolo11n.py"),
            "--task", "detect",
            "--data", str(self.data_root / "processed" / "dataset.yaml"),
            "--epochs", "100",  # Reduced for quick start
            "--batch", "16",
            "--imgsz", "1000",
            "--name", "detection_quickstart"
        ]
        
        try:
            logger.info("This may take 2-4 hours depending on your hardware...")
            result = subprocess.run(cmd, check=True)
            logger.info("✓ Detection training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Detection training failed: {e}")
            return False
    
    def run_segmentation_training(self):
        """Run segmentation model training."""
        logger.info("Starting segmentation model training...")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "train_yolo11n.py"),
            "--task", "segment", 
            "--data", str(self.data_root / "processed" / "dataset.yaml"),
            "--epochs", "100",  # Reduced for quick start
            "--batch", "8",
            "--imgsz", "1000",
            "--name", "segmentation_quickstart"
        ]
        
        try:
            logger.info("This may take 3-6 hours depending on your hardware...")
            result = subprocess.run(cmd, check=True)
            logger.info("✓ Segmentation training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Segmentation training failed: {e}")
            return False
    
    def run_severity_training(self):
        """Run severity estimation training."""
        logger.info("Starting severity estimation training...")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "severity_estimation.py"),
            "--data_dir", str(self.data_root / "processed"),
            "--model_dir", str(self.models_dir / "severity"),
            "--epochs", "30",  # Reduced for quick start
            "--batch_size", "32"
        ]
        
        try:
            logger.info("This may take 30-60 minutes...")
            result = subprocess.run(cmd, check=True)
            logger.info("✓ Severity training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Severity training failed: {e}")
            return False
    
    def run_evaluation(self):
        """Run model evaluation."""
        logger.info("Starting model evaluation...")
        
        cmd = [
            sys.executable,
            str(self.scripts_dir / "evaluate_models_simple.py"),
            "--output_dir", str(self.project_root / "evaluation" / "reports")
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            logger.info("✓ Model evaluation completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Model evaluation failed: {e}")
            return False
    
    def run_demo(self):
        """Launch the web demo application."""
        logger.info("Launching web demo...")
        
        # Check if streamlit is available
        try:
            import streamlit
        except ImportError:
            logger.error("Streamlit not installed. Please run: pip install streamlit")
            return False
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(self.project_root / "app" / "streamlit_app.py"),
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        try:
            logger.info("Starting Streamlit app on http://localhost:8501")
            logger.info("Press Ctrl+C to stop the demo")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Demo launch failed: {e}")
            return False
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")
            return True
    
    def run_full_pipeline(self):
        """Run the complete pipeline from start to finish."""
        logger.info("Starting complete pipeline...")
        
        steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Data Preprocessing", self.run_preprocessing),
            ("Detection Training", self.run_detection_training),
            ("Segmentation Training", self.run_segmentation_training),
            ("Severity Training", self.run_severity_training),
            ("Model Evaluation", self.run_evaluation)
        ]
        
        start_time = time.time()
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Pipeline failed at step: {step_name}")
                return False
            logger.info(f"✓ {step_name} completed")
        
        total_time = time.time() - start_time
        logger.info(f"✓ Complete pipeline finished in {total_time/3600:.1f} hours")
        
        # Optionally launch demo
        response = input("\\nWould you like to launch the web demo? (y/n): ")
        if response.lower() in ['y', 'yes']:
            self.run_demo()
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Car Damage Analysis Main Entry Point')
    parser.add_argument('--mode', type=str, 
                       choices=['preprocessing', 'training', 'evaluation', 'demo', 'full', 'all'],
                       default='full',
                       help='Pipeline mode to run (all is alias for full)')
    parser.add_argument('--check', action='store_true',
                       help='Only check prerequisites')
    
    args = parser.parse_args()
    
    pipeline = QuickStartPipeline()
    
    if args.check:
        success = pipeline.check_prerequisites()
        sys.exit(0 if success else 1)
    
    if args.mode == 'preprocessing':
        success = pipeline.check_prerequisites() and pipeline.run_preprocessing()
    elif args.mode == 'training':
        success = (pipeline.check_prerequisites() and 
                  pipeline.run_detection_training() and
                  pipeline.run_segmentation_training() and
                  pipeline.run_severity_training())
    elif args.mode == 'evaluation':
        success = pipeline.check_prerequisites() and pipeline.run_evaluation()
    elif args.mode == 'demo':
        success = pipeline.check_prerequisites() and pipeline.run_demo()
    elif args.mode == 'full' or args.mode == 'all':
        success = pipeline.run_full_pipeline()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False
    
    if success:
        logger.info("🎉 Pipeline completed successfully!")
        if args.mode != 'demo':
            print("\\n" + "="*60)
            print("NEXT STEPS:")
            print("="*60)
            print("1. Review the evaluation results in evaluation/reports/")
            print("2. Launch the web demo: python main.py --mode demo")
            print("3. Check model outputs in models/ directory")
            print("4. Read TRAINING.md for detailed documentation")
            print("="*60)
    else:
        logger.error("❌ Pipeline failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
