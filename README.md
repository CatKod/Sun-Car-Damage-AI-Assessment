# Car Damage AI Assessment System

A comprehensive computer vision pipeline using YOLOv11n for automated car damage detection, segmentation, and severity assessment.

## Overview

This system provides:
- **Object Detection**: Localize car damage using YOLOv11n
- **Instance Segmentation**: Generate precise damage masks
- **Severity Estimation**: Classify/regress damage severity levels
- **Comprehensive Evaluation**: mAP, IoU, accuracy metrics
- **Robustness Analysis**: Performance by shooting angle and completeness
- **Interactive Web Demo**: Real-time inference and visualization

## Project Structure

```
├── data/
│   ├── CarDD_COCO/                 # COCO format dataset
│   │   ├── annotations/            # COCO annotations and metadata
│   │   ├── CarDD-TR/              # Training images
│   │   ├── CarDD-VAL/             # Validation images
│   │   └── CarDD-TE/              # Test images
│   └── processed/                  # YOLO format processed data
├── models/
│   ├── detection/                  # Trained detection models
│   ├── segmentation/              # Trained segmentation models
│   └── severity/                  # Trained severity models
├── scripts/
│   ├── coco_data_preprocessing.py  # COCO to YOLO conversion
│   ├── train_yolov11n.py          # Training pipeline
│   ├── severity_estimation.py     # Severity model training
│   └── evaluate_models.py         # Comprehensive evaluation
├── evaluation/
│   ├── metrics.py                 # Custom evaluation metrics
│   └── reports/                   # Generated evaluation reports
├── web_app/
│   ├── backend/                   # FastAPI backend
│   ├── frontend/                  # Streamlit frontend
│   └── utils/                     # Utilities
└── app/
    └── streamlit_app.py           # Simple web interface
```

## Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 🚀 One-Command Pipeline

For the fastest start, use the main entry point:

```bash
# Check prerequisites
python main.py --check

# Run specific pipeline stages
python main.py --mode preprocessing    # Prepare dataset
python main.py --mode training         # Train all models  
python main.py --mode evaluation       # Evaluate models
python main.py --mode demo            # Launch web app

# Run complete pipeline (data → training → evaluation)
python main.py --mode all
```

### 2. Check Training Status

Before starting or resuming training, check what's been completed:

```bash
python check_status.py
```

This will show you:
- ✓ What stages are complete
- ✗ What stages are pending  
- 📋 Next recommended action

### 3. Resume Interrupted Training

If your training was interrupted, you can resume from where it left off:

```bash
# Resume from evaluation stage (after model training completed)
python resume_training.py --stage evaluation

# Resume from robustness analysis stage
python resume_training.py --stage robustness

# Resume from export stage only
python resume_training.py --stage export

# Use a specific model (if auto-detection fails)
python resume_training.py --stage evaluation --model-path "runs/detect/train/weights/best.pt"
```

### 4. Run Individual Phases

You can also run individual phases independently:

```bash
# Run only evaluation
python run_phase.py evaluation

# Run only robustness analysis  
python run_phase.py robustness

# Run only model export
python run_phase.py export

# Generate summary report
python run_phase.py summary
```

### 5. Complete Training (if starting fresh)

```bash
# Prepare dataset
python data/coco_data_preprocessing.py

# Train detection model only (recommended for laptops)
python run_detection_only.py

# OR run full pipeline (requires more resources)
python scripts/train_yolo11n.py
```

### 6. Run the Streamlit Web Application

**Option A: Use the main script**
```bash
python main.py --mode demo
```

**Option B: Use the provided scripts**

On Windows:
```cmd
run_app.bat
```

On Linux/Mac:
```bash
chmod +x run_app.sh
./run_app.sh
```

**Option C: Run directly**
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Then open your browser and go to: http://localhost:8501

### 7. Using the Web App

1. **Model Selection**: The app automatically loads the best available trained model
   - Switch between available models in the sidebar
   - Upload your own custom YOLO models (.pt files)

2. **Single Image Analysis**:
   - Upload a vehicle image (PNG, JPG, JPEG)
   - Click "Analyze Image" to detect damage
   - View detailed results with confidence scores

3. **Batch Processing**:
   - Upload multiple images for batch analysis
   - Download results in JSON format
   - View comprehensive statistics and charts

4. **Configuration**:
   - Adjust confidence threshold for detections
   - Modify IoU threshold for Non-Maximum Suppression
   - Configure advanced settings in the sidebar

### 8. Train Your Own Models

```bash
# Train detection and segmentation
python scripts/train_yolov11n.py

# Train severity estimation
python scripts/severity_estimation.py
```

## Features

### Data Pipeline
- COCO to YOLO format conversion
- Multi-modal processing (original, edge, mask images)
- Automatic severity estimation from damage area
- Comprehensive dataset analysis and visualization

### Model Training
- YOLOv11n for detection and segmentation
- Advanced augmentation and optimization
- Multi-metric evaluation (mAP@0.5, mAP@0.5:0.95, IoU)
- Robustness analysis by metadata attributes

### Web Interface
- Real-time damage detection and visualization
- Interactive mask overlay and confidence adjustment
- Damage severity assessment and reporting
- Export results in JSON/PNG formats

### Evaluation & Analysis
- Per-category performance metrics
- Robustness testing by shooting angle and completeness
- Failure case analysis and visualization
- Automated report generation

## Model Performance

| Model | Task | mAP@0.5 | mAP@0.5:0.95 | Inference Time |
|-------|------|---------|--------------|----------------|
| YOLOv11n | Detection | 0.XX | 0.XX | XX ms |
| YOLOv11n | Segmentation | 0.XX | 0.XX | XX ms |
| Severity | Classification | XX% | - | XX ms |

## Dataset Information

- **Total Images**: Training, validation, and test splits
- **Damage Categories**: Various types of car damage
- **Annotations**: Bounding boxes and segmentation masks
- **Metadata**: Shooting angle, completeness, severity levels

## Citation

If you use this work, please cite:
```bibtex
@software{car_damage_ai_assessment,
  title={Car Damage AI Assessment System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

MIT License - see LICENSE file for details.
