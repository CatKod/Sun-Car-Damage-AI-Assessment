"""
Car Damage Analysis - yolo11n Training Pipeline
===============================================

This script implements a comprehensive training pipeline for yolo11n
on car damage detection and segmentation tasks.

Features:
- yolo11n detection training
- Instance segmentation extension
- Multi-modal training (Image + Edge)
- Advanced augmentation strategies
- Comprehensive evaluation metrics
- Model export and optimization

Author: AI Engineer
Date: July 2025
"""

import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, SegmentMetrics
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import wandb
from sklearn.metrics import confusion_matrix, classification_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarDamageYOLOv11Trainer:
    """
    Advanced yolo11n trainer for car damage detection and segmentation
    """
    
    def __init__(self, 
                 dataset_path: str = "yolo_dataset",
                 model_name: str = "models/yolo11n.pt",
                 project_name: str = "car_damage_yolo11n",
                 use_wandb: bool = False):
        """
        Initialize the yolo11n trainer
        
        Args:
            dataset_path: Path to YOLO dataset
            model_name: yolo11n model variant
            project_name: Project name for experiments
            use_wandb: Whether to use Weights & Biases logging
        """
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.project_name = project_name
        self.use_wandb = use_wandb
        
        # Load dataset configuration
        self.config_path = self.dataset_path / 'dataset.yaml'
        with open(self.config_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        # Load dataset info
        info_path = self.dataset_path / 'dataset_info.json'
        with open(info_path, 'r') as f:
            self.dataset_info = json.load(f)
        
        self.model = None
        self.results = None
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f'runs/{project_name}_{timestamp}')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Weights & Biases if requested
        if self.use_wandb:
            wandb.init(
                project=f"car-damage-{project_name}",
                name=f"{project_name}_{timestamp}",
                config={
                    "model": model_name,
                    "dataset": str(dataset_path),
                    "classes": self.dataset_config['nc']
                }
            )
        
        logger.info(f"Initialized yolo11n trainer")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Classes: {self.dataset_config['nc']}")
        logger.info(f"Results: {self.results_dir}")
    
    def setup_device_and_memory(self):
        """Setup optimal device and memory configuration"""
        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            
            logger.info(f"Using GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Adjust batch size based on GPU memory
            if gpu_memory < 4:
                self.batch_size = 4
                self.workers = 2
            elif gpu_memory < 8:
                self.batch_size = 8
                self.workers = 4
            else:
                self.batch_size = 16
                self.workers = 8
        else:
            self.device = 'cpu'
            self.batch_size = 2
            self.workers = 2
            logger.info("Using CPU (GPU not available)")
        
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Workers: {self.workers}")
    
    def create_training_config(self) -> Dict:
        """
        Create training configuration for yolo11n detection
        
        Returns:
            Training configuration dictionary
        """
        base_config = {
            # Model and data
            'model': self.model_name,
            'data': str(self.config_path),
            'task': 'detect',
            'patience': 20, 

            # Training parameters
            'epochs': 150,
            'batch': 1,
            'imgsz': 1000,
            'device': self.device,
            'workers': 0,
            
            # Project settings
            'project': str(self.results_dir.parent),
            'name': self.results_dir.name,
            'exist_ok': True,
            
            # Optimization
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss weights
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Data augmentation
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.3,
            
            # Validation and saving
            'val': True,
            'save': True,
            'save_period': 10,
            'cache': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'dropout': 0.0,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'plots': False,
            'show_labels': True,
            'show_conf': True,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'embed': None,
        }
        
        return base_config
    
    def load_model(self, task: str = 'detect'):
        """Load yolo11n model for detection only"""
        logger.info(f"Loading yolo11n model for detection task...")
        
        try:
            self.model = YOLO(self.model_name)
            logger.info(f"Detection model loaded successfully: {self.model.model}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train_detection_model(self) -> Dict:
        """Train yolo11n for damage detection"""
        logger.info("Starting detection model training...")
        
        # Load detection model
        self.load_model('detect')
        
        # Create training configuration
        train_config = self.create_training_config()
        
        try:
            # Start training
            results = self.model.train(**train_config)
            
            # Save detection model - use best.pt directly
            # Save detection model - use best.pt directly
            best_path = self.results_dir / 'weights' / 'best.pt'
            detection_model_path = self.results_dir / 'weights' / f"{Path(self.model_name).stem}_best.pt"
            if best_path.exists():
                import shutil
                shutil.copy2(best_path, detection_model_path)
            logger.info(f"Detection model saved at: {detection_model_path}")
            
            logger.info("Detection training completed!")
            return results
        except Exception as e:
            logger.error(f"Detection training failed: {e}")
            raise
    
    def evaluate_model(self, model_path: str) -> Dict:
        """
        Comprehensive model evaluation for detection
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info(f"Evaluating detection model...")
        
        # Load trained model
        model = YOLO(model_path)
        
        # Run validation
        val_results = model.val(
            data=str(self.config_path),
            split='val',
            save_json=True,
            conf=0.25,
            iou=0.6,
            max_det=300,
            plots=True
        )

        #clear cache to avoid memory issues
        torch.cuda.empty_cache()
        
        # Run test evaluation
        test_results = model.val(
            data=str(self.config_path),
            split='test',
            save_json=True,
            conf=0.25,
            iou=0.6,
            max_det=300,
            plots=True
        )
        
        # Calculate detailed metrics
        metrics = self.calculate_detailed_metrics(val_results, test_results)
        
        return metrics
    
    def calculate_detailed_metrics(self, val_results, test_results) -> Dict:
        """Calculate comprehensive evaluation metrics for detection only"""
        
        def extract_metrics(results, split_name):
            if hasattr(results, 'results_dict'):
                metrics_dict = results.results_dict
            else:
                metrics_dict = {}
            
            base_metrics = {
                f'{split_name}_mAP50': metrics_dict.get('metrics/mAP50(B)', 0),
                f'{split_name}_mAP50-95': metrics_dict.get('metrics/mAP50-95(B)', 0),
                f'{split_name}_precision': metrics_dict.get('metrics/precision(B)', 0),
                f'{split_name}_recall': metrics_dict.get('metrics/recall(B)', 0),
            }
            
            # Calculate F1 score
            precision = base_metrics[f'{split_name}_precision']
            recall = base_metrics[f'{split_name}_recall']
            if precision + recall > 0:
                base_metrics[f'{split_name}_f1'] = 2 * (precision * recall) / (precision + recall)
            else:
                base_metrics[f'{split_name}_f1'] = 0
            
            return base_metrics
        
        # Extract metrics for both splits
        metrics = {}
        metrics.update(extract_metrics(val_results, 'val'))
        metrics.update(extract_metrics(test_results, 'test'))
        
        # Add model info
        metrics.update({
            'model_task': 'detect',
            'model_name': self.model_name,
            'dataset_classes': self.dataset_config['nc'],
            'training_date': datetime.now().isoformat()
        })
        
        return metrics
    
    def analyze_robustness(self, model_path: str) -> Dict:
        """
        Analyze model robustness across different conditions
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Robustness analysis results
        """
        logger.info("Analyzing model robustness...")
        
        # Load model
        model = YOLO(model_path)
        
        # Load metadata for analysis
        metadata_df = pd.read_csv(self.dataset_path.parent / "data" / "CarDD_COCO" / "annotations" / "image_info.csv")
        
        # Analysis by shooting angle
        angle_results = {}
        for angle in metadata_df['shooting angle'].unique():
            angle_images = metadata_df[metadata_df['shooting angle'] == angle]['file_name'].tolist()
            angle_results[angle] = self.evaluate_subset(model, angle_images)
        
        # Analysis by completeness
        completeness_results = {}
        for completeness in metadata_df['complete or partial'].unique():
            complete_images = metadata_df[metadata_df['complete or partial'] == completeness]['file_name'].tolist()
            completeness_results[completeness] = self.evaluate_subset(model, complete_images)
        
        robustness_analysis = {
            'angle_performance': angle_results,
            'completeness_performance': completeness_results,
            'overall_consistency': self.calculate_consistency_metrics(angle_results, completeness_results)
        }
        
        # Save robustness report
        self.save_robustness_report(robustness_analysis)
        
        return robustness_analysis
    
    def evaluate_subset(self, model, image_list: List[str]) -> Dict:
        """Evaluate model on a subset of images"""
        try:
            total_predictions = 0
            sampled_images = image_list[:10]  # Sample for efficiency
            
            for image_name in sampled_images:
                image_path = self.dataset_path / 'test' / 'images' / image_name
                if image_path.exists():
                    results = model.predict(str(image_path), verbose=False)
                    if results and results[0].boxes is not None:
                        total_predictions += len(results[0].boxes)

            num_sampled = len(sampled_images)
            return {
                'num_images': len(image_list),
                'avg_detections': total_predictions / num_sampled if num_sampled > 0 else 0,
                'subset_performance': 'evaluated'
            }
        except Exception as e:
            logger.warning(f"Subset evaluation failed: {e}")
            return {'error': str(e)}
    
    def calculate_consistency_metrics(self, angle_results: Dict, completeness_results: Dict) -> Dict:
        """Calculate consistency metrics across different conditions"""
        
        # Extract performance values
        angle_perfs = [result.get('avg_detections', 0) for result in angle_results.values() if 'error' not in result]
        complete_perfs = [result.get('avg_detections', 0) for result in completeness_results.values() if 'error' not in result]
        
        consistency = {
            'angle_variance': np.var(angle_perfs) if angle_perfs else 0,
            'completeness_variance': np.var(complete_perfs) if complete_perfs else 0,
            'overall_stability': 'high' if (np.var(angle_perfs) < 0.1 and np.var(complete_perfs) < 0.1) else 'moderate'
        }
        
        return consistency
    
    def save_robustness_report(self, analysis: Dict):
        """Save robustness analysis report"""
        report_path = self.results_dir / 'robustness_analysis.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Robustness analysis saved to {report_path}")
    
    def export_model(self, model_path: str):
        """Export trained detection model to different formats"""
        logger.info("Exporting detection model to different formats...")
        
        export_dir = self.results_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        formats = ['onnx', 'torchscript']  # Removed coreml for compatibility
        
        if Path(model_path).exists():
            model = YOLO(model_path)
            
            for format_name in formats:
                try:
                    logger.info(f"Exporting detection model to {format_name}...")
                    exported_path = model.export(
                        format=format_name,
                        imgsz=1000,
                        optimize=True,
                        half=False
                    )
                    logger.info(f"Exported detection model to: {exported_path}")
                except Exception as e:
                    logger.warning(f"Failed to export to {format_name}: {e}")
    
    def create_training_summary(self, detection_metrics: Dict, robustness_analysis: Dict):
        """Create comprehensive training summary for detection only"""
        
        summary = {
            'training_info': {
                'model': self.model_name,
                'dataset': str(self.dataset_path),
                'project': self.project_name,
                'training_date': datetime.now().isoformat(),
                'device': self.device,
                'batch_size': self.batch_size
            },
            'detection_performance': detection_metrics,
            'robustness_analysis': robustness_analysis,
            'model_paths': {
                'detection': str(self.results_dir / 'weights' / 'best.pt')
            }
        }
        
        # Save summary
        summary_path = self.results_dir / 'training_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training summary saved to {summary_path}")
        
        # Create markdown report
        self.create_markdown_report(summary)
    
    def create_markdown_report(self, summary: Dict):
        """Create markdown training report for detection only"""
        
        report = f"""# Car Damage Analysis - yolo11n Training Report

## Training Configuration
- **Model**: {summary['training_info']['model']}
- **Dataset**: {summary['training_info']['dataset']}
- **Device**: {summary['training_info']['device']}
- **Batch Size**: {summary['training_info']['batch_size']}
- **Training Date**: {summary['training_info']['training_date']}

## Detection Performance
"""
        
        detection_metrics = summary['detection_performance']
        report += f"""
- **Validation mAP@0.5**: {detection_metrics.get('val_mAP50', 0):.4f}
- **Validation mAP@0.5:0.95**: {detection_metrics.get('val_mAP50-95', 0):.4f}
- **Test mAP@0.5**: {detection_metrics.get('test_mAP50', 0):.4f}
- **Test mAP@0.5:0.95**: {detection_metrics.get('test_mAP50-95', 0):.4f}
- **Test Precision**: {detection_metrics.get('test_precision', 0):.4f}
- **Test Recall**: {detection_metrics.get('test_recall', 0):.4f}
- **Test F1**: {detection_metrics.get('test_f1', 0):.4f}

## Robustness Analysis
- **Overall Stability**: {summary['robustness_analysis']['overall_consistency']['overall_stability']}
- **Angle Variance**: {summary['robustness_analysis']['overall_consistency']['angle_variance']:.4f}
- **Completeness Variance**: {summary['robustness_analysis']['overall_consistency']['completeness_variance']:.4f}

## Model Output
- **Detection Model**: `{summary['model_paths']['detection']}`
"""
        
        report_path = self.results_dir / 'training_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Training report saved to {report_path}")
    
    def run_complete_training_pipeline(self):
        """Run the complete detection training pipeline (no segmentation)"""
        logger.info("Starting yolo11n detection training pipeline...")
        
        try:
            # Setup
            self.setup_device_and_memory()
            
            # Train detection model only
            logger.info("Phase 1: Detection Model Training")
            detection_results = self.train_detection_model()
            detection_model_path = self.results_dir / 'weights' / 'best.pt'
            
            # Evaluate detection model
            logger.info("Phase 2: Detection Model Evaluation")
            detection_metrics = self.evaluate_model(str(detection_model_path))
            
            # Robustness analysis
            logger.info("Phase 3: Robustness Analysis")
            robustness_analysis = self.analyze_robustness(str(detection_model_path))
            
            # Export model
            logger.info("Phase 4: Model Export")
            self.export_model(str(detection_model_path))
            
            # Create final summary
            logger.info("Phase 5: Creating Summary")
            self.create_training_summary(detection_metrics, robustness_analysis)
            
            logger.info("Detection training pipeline finished successfully!")
            
            return {
                'detection_metrics': detection_metrics,
                'robustness_analysis': robustness_analysis,
                'model_path': str(detection_model_path)
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()


def main():
    """Main execution function"""
    # Configuration
    dataset_path = "yolo_dataset"
    model_name = "models/yolo11n.pt"
    project_name = "car_damage_yolo11n"
    use_wandb = False  # Set to True to enable W&B logging
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        logger.error(f"Dataset path {dataset_path} does not exist!")
        logger.info("Please run coco_data_preprocessing.py first to create the YOLO dataset.")
        return
    
    # Initialize trainer
    trainer = CarDamageYOLOv11Trainer(
        dataset_path=dataset_path,
        model_name=model_name,
        project_name=project_name,
        use_wandb=use_wandb
    )
    
    # Run complete training pipeline
    results = trainer.run_complete_training_pipeline()
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {trainer.results_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
