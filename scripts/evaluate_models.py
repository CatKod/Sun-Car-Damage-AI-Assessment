import os
import json
import logging
from pathlib import Path
from ultralytics import YOLO
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_model():
    """Find the latest trained model."""
    runs_dir = Path("runs")
    if runs_dir.exists():
        # Check for custom format first
        custom_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "car_damage" in d.name]
        if custom_dirs:
            latest_dir = max(custom_dirs, key=lambda d: d.stat().st_mtime)
            weights_dir = latest_dir / "weights"
            if weights_dir.exists():
                for model_name in ["detection_best.pt", "best.pt", "last.pt"]:
                    model_path = weights_dir / model_name
                    if model_path.exists():
                        logger.info(f"Found model: {model_path}")
                        return str(model_path)
    
    logger.warning("No trained model found, using default")
    return "models/yolo11n.pt"

def run_yolo_evaluation(model_path, data_path, output_dir):
    """Run YOLO's built-in evaluation."""
    logger.info(f"Running evaluation with model: {model_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Run validation on val split
        logger.info("Evaluating on validation split...")
        val_results = model.val(
            data=data_path,
            split='val',
            save_json=True,
            conf=0.001,
            iou=0.6,
            plots=True,
            save_dir=output_dir / 'val_results'
        )
        
        # Run validation on test split
        logger.info("Evaluating on test split...")
        test_results = model.val(
            data=data_path,
            split='test', 
            save_json=True,
            conf=0.001,
            iou=0.6,
            plots=True,
            save_dir=output_dir / 'test_results'
        )
        
        # Extract metrics
        val_metrics = {}
        test_metrics = {}
        
        if hasattr(val_results, 'results_dict'):
            val_dict = val_results.results_dict
            val_metrics = {
                'mAP50': val_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': val_dict.get('metrics/mAP50-95(B)', 0),
                'precision': val_dict.get('metrics/precision(B)', 0),
                'recall': val_dict.get('metrics/recall(B)', 0)
            }
            val_metrics['f1'] = (2 * val_metrics['precision'] * val_metrics['recall'] / 
                               (val_metrics['precision'] + val_metrics['recall']) 
                               if (val_metrics['precision'] + val_metrics['recall']) > 0 else 0)
        
        if hasattr(test_results, 'results_dict'):
            test_dict = test_results.results_dict
            test_metrics = {
                'mAP50': test_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': test_dict.get('metrics/mAP50-95(B)', 0),
                'precision': test_dict.get('metrics/precision(B)', 0),
                'recall': test_dict.get('metrics/recall(B)', 0)
            }
            test_metrics['f1'] = (2 * test_metrics['precision'] * test_metrics['recall'] / 
                                (test_metrics['precision'] + test_metrics['recall']) 
                                if (test_metrics['precision'] + test_metrics['recall']) > 0 else 0)
        
        # Create summary report
        report = {
            'model_path': model_path,
            'data_path': data_path,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        # Save report
        with open(output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown report
        markdown_content = f"""# Detection Model Evaluation Report

## Model Information
- **Model Path:** {model_path}
- **Data Path:** {data_path}

## Validation Set Results
- **mAP@0.5:** {val_metrics.get('mAP50', 0):.3f}
- **mAP@0.5:0.95:** {val_metrics.get('mAP50-95', 0):.3f}
- **Precision:** {val_metrics.get('precision', 0):.3f}
- **Recall:** {val_metrics.get('recall', 0):.3f}
- **F1 Score:** {val_metrics.get('f1', 0):.3f}

## Test Set Results
- **mAP@0.5:** {test_metrics.get('mAP50', 0):.3f}
- **mAP@0.5:0.95:** {test_metrics.get('mAP50-95', 0):.3f}
- **Precision:** {test_metrics.get('precision', 0):.3f}
- **Recall:** {test_metrics.get('recall', 0):.3f}
- **F1 Score:** {test_metrics.get('f1', 0):.3f}
"""
        
        with open(output_dir / 'evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Evaluation completed! Results saved to: {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("DETECTION MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"\nValidation Set:")
        print(f"  mAP@0.5: {val_metrics.get('mAP50', 0):.3f}")
        print(f"  Precision: {val_metrics.get('precision', 0):.3f}")
        print(f"  Recall: {val_metrics.get('recall', 0):.3f}")
        print(f"  F1 Score: {val_metrics.get('f1', 0):.3f}")
        print(f"\nTest Set:")
        print(f"  mAP@0.5: {test_metrics.get('mAP50', 0):.3f}")
        print(f"  Precision: {test_metrics.get('precision', 0):.3f}")
        print(f"  Recall: {test_metrics.get('recall', 0):.3f}")
        print(f"  F1 Score: {test_metrics.get('f1', 0):.3f}")
        print("="*60)
        
        return report
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Detection Model Evaluation')
    parser.add_argument('--config', type=str, default='configs/evaluation_config.yaml',
                       help='Path to evaluation configuration file')
    parser.add_argument('--output_dir', type=str, default='evaluation/reports',
                       help='Output directory for evaluation results')
    parser.add_argument('--model', type=str, 
                       help='Path to detection model (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Load or create config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Create default config
        model_path = args.model or find_latest_model()
        config = {
            'detection_model': model_path,
            'data_root': 'yolo_dataset',
            'output_dir': args.output_dir
        }
        
        # Save config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Created configuration: {config_path}")
    
    # Get paths
    model_path = args.model or config.get('detection_model') or find_latest_model()
    data_path = Path(config.get('data_root', 'yolo_dataset')) / 'dataset.yaml'
    output_dir = config.get('output_dir', args.output_dir)
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    # Check if data config exists
    if not data_path.exists():
        logger.error(f"Dataset config not found: {data_path}")
        return 1
    
    # Run evaluation
    try:
        report = run_yolo_evaluation(model_path, str(data_path), output_dir)
        logger.info("Evaluation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
