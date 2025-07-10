# Car Damage Analysis System - yolo11n Pipeline
## 🏗️ Project Architecture

### System Components

```
car-damage-analysis/
├── 📊 data/
│   ├── image_info.csv              # Metadata
│   ├── instances_train2017.json     # COCO training annotations
│   ├── instances_val2017.json       # COCO validation annotations
│   ├── instances_test2017.json      # COCO test annotations
│   ├── images/                      # Original images
│   ├── edges/                       # Edge-detected images
│   └── masks/                       # Segmentation ground truth
│
├── 🧠 models/
│   ├── yolo11n_detection.pt       # Detection model
│   ├── yolo11n_segmentation.pt    # Segmentation model
│   ├── severity_classifier.pkl     # Severity estimation model
│   └── model_configs/               # Training configurations
│
├── 🔧 scripts/
│   ├── data_preprocessing.py        # COCO to YOLO conversion
│   ├── train_detection.py          # yolo11n detection training
│   ├── train_segmentation.py       # Segmentation training
│   ├── severity_estimation.py      # Severity model training
│   ├── evaluation.py               # Model evaluation
│   └── inference.py                # Inference pipeline
│
├── 🌐 web_app/
│   ├── backend/
│   │   ├── app.py                   # FastAPI backend
│   │   ├── models.py                # Pydantic models
│   │   └── inference_engine.py     # Model inference
│   ├── frontend/
│   │   ├── streamlit_app.py         # Streamlit frontend
│   │   └── components/              # UI components
│   └── static/                      # Static assets
│
├── 📈 evaluation/
│   ├── metrics.py                   # Evaluation metrics
│   ├── visualizations.py           # Result visualizations
│   └── reports/                     # Evaluation reports
│
└── 📋 configs/
    ├── training_config.yaml         # Training parameters
    ├── model_config.yaml            # Model architecture
    └── evaluation_config.yaml       # Evaluation settings
```

### Pipeline Flow

1. **Data Preprocessing** → COCO to YOLO conversion
2. **Model Training** → Detection + Segmentation + Severity
3. **Evaluation** → mAP, IoU, Accuracy metrics
4. **Web Application** → Interactive demo
5. **Deployment** → Model serving and inference

### Key Features

✅ **yolo11n Detection** - Bounding box prediction
✅ **Instance Segmentation** - Pixel-level damage masks
✅ **Severity Estimation** - Minor/Moderate/Severe classification
✅ **Multi-modal Analysis** - Original + Edge + Mask images
✅ **Robustness Testing** - Angle and completeness analysis
✅ **Interactive Web UI** - Real-time inference demo
✅ **Comprehensive Evaluation** - Full metrics and visualizations
