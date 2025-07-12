# Car Damage AI Assessment System

## Author Information
- **Name**: Hoàng Kim Vĩnh
- **Student ID**: 20235876
- **University**: Hanoi University of Science and Technology (HUST)

## Project Overview

A comprehensive computer vision pipeline using YOLOv11n for automated car damage detection, segmentation, and severity assessment. This system provides an end-to-end solution for analyzing car damage through AI-powered image processing, offering real-time detection capabilities with high accuracy.

### Main Features Implemented
- **Object Detection**: Localize car damage using YOLOv11n
- **Instance Segmentation**: Generate precise damage masks
- **Severity Estimation**: Classify/regress damage severity levels
- **Comprehensive Evaluation**: mAP, IoU, accuracy metrics
- **Robustness Analysis**: Performance by shooting angle and completeness
- **Interactive Web Demo**: Real-time inference and visualization

## Technologies Used

### Backend
- **Python 3.11**: Main programming language for AI model development
- **PyTorch**: Deep learning framework for model training and inference
- **Ultralytics YOLOv11n**: State-of-the-art object detection and segmentation model
  - *Reason*: Excellent balance between accuracy and inference speed, perfect for real-time applications
- **OpenCV**: Computer vision library for image processing
- **NumPy & Pandas**: Data manipulation and numerical computations
- **Streamlit**: Web application framework for the demo interface
  - *Reason*: Rapid prototyping and easy deployment of ML models with minimal code

### Frontend (Web Interface)
- **Streamlit**: Interactive web application framework
- **Plotly**: Interactive visualization library for charts and graphs
- **PIL (Python Imaging Library)**: Image processing and display
- **Base64 encoding**: For image data transfer and display in web interface

### Data Processing & Evaluation
- **COCO dataset format**: Standard annotation format for object detection
- **YOLO format conversion**: Custom preprocessing pipeline
- **JSON/YAML**: Configuration and metadata storage
- **Matplotlib/Seaborn**: Data visualization and analysis

### Model Training & Optimization
- **CUDA**: GPU acceleration for training (if available)
- **Weights & Biases integration**: Experiment tracking and monitoring
- **Model checkpointing**: Automatic saving of best models during training

## Project Structure

### Backend Components
```
Sun-Car-Damage-AI-Assessment/
├── 📁 scripts/                              # Core AI pipeline scripts
│   ├── coco_data_preprocessing.py           # Data preprocessing and conversion
│   ├── train_yolov11n.py                    # Model training pipeline
│   └── evaluate_models.py                   # Model evaluation and metrics
├── 📁 models/                               # Trained model artifacts
│   ├── 📁 detection/                        # Object detection models
│   ├── 📁 segmentation/                     # Instance segmentation models
│   └── 📁 exports/                          # ONNX exported models
├── 📁 evaluation/                           # Evaluation framework
│   ├── metrics.py                           # Custom metrics implementation
│   └── 📁 reports/                          # Performance analysis reports
├── 📁 configs/                              # Configuration management
│   ├── training_config.yaml                # Training hyperparameters
│   ├── model_config.yaml                   # Model architecture settings
│   └── evaluation_config.yaml              # Evaluation parameters
└── main.py                                  # Main execution entry point
```

### Frontend Components
```
├── 📁 app/                                  # Web application interface
│   ├── streamlit_app.py                    # Main Streamlit application
│   └── requirements.txt                    # Frontend dependencies
├── 📁 screenshots/                         # Demo screenshots and videos
│   ├── Demo_web.mp4                        # Application demo video
│   └── image.png                           # Interface screenshots
├── run_app.bat                             # Windows application launcher
└── run_app.sh                              # Unix/Linux application launcher
```

### Data Management
```
├── 📁 data/                                 # Dataset storage and management
│   └── 📁 CarDD_COCO/                      # Original COCO format dataset
│       ├── 📁 annotations/                 # Annotation files (JSON)
│       ├── 📁 CarDD-TR/                    # Training images (3000+)
│       ├── 📁 CarDD-VAL/                   # Validation images (800+)
│       └── 📁 CarDD-TE/                    # Test images (800+)
├── 📁 yolo_dataset/                        # Processed YOLO format data
│   ├── 📁 train/, 📁 val/, 📁 test/        # Split datasets with images & labels
│   ├── dataset.yaml                        # YOLO configuration
│   └── 📁 analysis/                        # Dataset analysis results
└── 📁 runs/                                # Training experiment logs
```

### 🏷️ Damage Class Mapping

The YOLO dataset uses the following class mapping for car damage detection:

| Class ID | Damage Type | Description | Color Code |
|----------|-------------|-------------|------------|
| 0 | **Dent** | Surface deformation without paint damage | 🟦 Blue |
| 1 | **Scratch** | Surface paint damage and scratches | 🟩 Green |
| 2 | **Crack** | Cracks in body panels or bumpers | 🟡 Yellow |
| 3 | **Glass** | Windshield or window damage | 🟣 Purple |
| 4 | **Smash** | Severe impact damage | 🟠 Orange |
| 5 | **Spacing** | Panel gaps and alignment issues | 🔴 Red |

### 📊 Dataset Statistics

Based on the sample label files analyzed:

- **Multi-class Images**: Many images contain multiple damage types
- **Dense Annotations**: Complex damage scenes with 8-12 bounding boxes
- **Damage Distribution**: 
  - Dents (Class 0): Most common damage type
  - Scratches (Class 1): Frequently co-occurs with dents
  - Smash (Class 4): Often appears with other damage types
  - Glass (Class 3): Less frequent but critical for safety assessment
- **Annotation Format**: YOLO format with normalized coordinates
  - `class_id x_center y_center width height`

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

|   Model  |    Task   | mAP@0.5 | mAP@0.5:0.95 | Inference Time |
|----------|-----------|---------|--------------|----------------|
| YOLOv11n | Detection | 0.7357  |    0.5833    | 34 ms          |


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
  author={Hoang Kim Vinh},
  year={2025},
  url={https://github.com/CatKod/Sun-Car-Damage-AI-Assessment}
}
```

