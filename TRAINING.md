# Car Damage Analysis Training Guide

This guide provides step-by-step instructions for training the complete car damage analysis system using yolo11n for detection, segmentation, and severity estimation.

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3070 or better recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space
- **CUDA**: Version 11.0 or higher
- **Python**: 3.8 or higher

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Sun-Car-Damage-AI-Assessment
```

2. **Create virtual environment:**
```bash
# Using conda (recommended)
conda create -n car-damage python=3.9
conda activate car-damage

# Or using venv
python -m venv car-damage-env
source car-damage-env/bin/activate  # Linux/Mac
# car-damage-env\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Preparation

### 1. Download Dataset
Place your CarDD COCO dataset in the following structure:
```
data/
├── CarDD_COCO/
│   ├── annotations/
│   │   ├── image_info.csv
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   └── instances_test2017.json
│   ├── CarDD-TR/     # Training images
│   ├── CarDD-VAL/    # Validation images
│   └── CarDD-TE/     # Test images
```

### 2. Data Preprocessing
Convert COCO format to YOLO format and extract severity labels:

```bash
python scripts/coco_data_preprocessing.py \
    --data_root data/CarDD_COCO \
    --output_root data/processed
```

**Expected Output:**
- YOLO format dataset in `data/processed/`
- Dataset analysis in `data/processed/analysis/`
- Severity labels in `data/processed/severity_labels.json`

### 3. Verify Data Processing
Check that the conversion was successful:
```bash
ls data/processed/
# Should show: train/, val/, test/, dataset.yaml, severity_labels.json, analysis/

# Check dataset configuration
cat data/processed/dataset.yaml
```

## Model Training

### 1. Detection Model Training

Train yolo11n for damage detection:

```bash
python scripts/train_yolo11n.py \
    --task detect \
    --data data/processed/dataset.yaml \
    --epochs 300 \
    --batch 16 \
    --imgsz 640 \
    --name detection_experiment
```

**Key Parameters:**
- `--task detect`: Training for object detection
- `--epochs 300`: Number of training epochs
- `--batch 16`: Batch size (adjust based on GPU memory)
- `--imgsz 640`: Input image size
- `--device 0`: GPU device (auto-detected by default)

**Expected Training Time:** 4-8 hours on RTX 3080

### 2. Segmentation Model Training

Train yolo11n for instance segmentation:

```bash
python scripts/train_yolo11n.py \
    --task segment \
    --data data/processed/dataset.yaml \
    --epochs 300 \
    --batch 8 \
    --imgsz 640 \
    --name segmentation_experiment
```

**Key Differences:**
- Smaller batch size (8 vs 16) due to higher memory requirements
- Longer training time due to mask prediction complexity

**Expected Training Time:** 6-12 hours on RTX 3080

### 3. Severity Estimation Training

Train the severity estimation model:

```bash
python scripts/severity_estimation.py \
    --data_dir data/processed \
    --model_dir models/severity \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

**Expected Training Time:** 1-2 hours on RTX 3080

## Training Monitoring

### 1. Monitor Training Progress

During training, monitor these files:
- `runs/detect/detection_experiment/` - Detection training logs
- `runs/segment/segmentation_experiment/` - Segmentation training logs
- `models/severity/` - Severity model checkpoints

### 2. Training Metrics to Watch

**Detection & Segmentation:**
- `mAP@0.5`: Should reach >0.75 for good performance
- `mAP@0.5:0.95`: Should reach >0.45 for good performance
- Training loss: Should decrease steadily
- Validation loss: Should decrease without overfitting

**Severity Estimation:**
- Validation accuracy: Should reach >85%
- Loss convergence: Both training and validation losses should stabilize

### 3. TensorBoard Visualization

View training progress in real-time:
```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in your browser
```

## Training Tips & Troubleshooting

### Common Issues and Solutions

**1. Out of Memory (OOM) Error:**
```bash
# Reduce batch size
python scripts/train_yolo11n.py --batch 8  # Instead of 16

# Reduce image size
python scripts/train_yolo11n.py --imgsz 480  # Instead of 640

# Enable mixed precision
python scripts/train_yolo11n.py --amp
```

**2. Slow Training:**
- Ensure CUDA is properly installed
- Use SSD storage for dataset
- Increase number of workers: `--workers 8`
- Enable data caching: `--cache ram`

**3. Poor Performance:**
- Increase training epochs
- Adjust learning rate: `--lr0 0.01` or `--lr0 0.0001`
- Use different optimizer: `--optimizer SGD`
- Enable data augmentation: Check `configs/training_config.yaml`

### Optimization Strategies

**1. Hyperparameter Tuning:**
```bash
# Learning rate experimentation
python scripts/train_yolo11n.py --lr0 0.01
python scripts/train_yolo11n.py --lr0 0.001
python scripts/train_yolo11n.py --lr0 0.0001

# Batch size optimization
python scripts/train_yolo11n.py --batch 8
python scripts/train_yolo11n.py --batch 16
python scripts/train_yolo11n.py --batch 32
```

**2. Advanced Training:**
```bash
# Multi-scale training
python scripts/train_yolo11n.py --rect

# Resume training from checkpoint
python scripts/train_yolo11n.py --resume runs/detect/train/weights/last.pt

# Transfer learning from custom checkpoint
python scripts/train_yolo11n.py --weights your_pretrained_model.pt
```

## Model Evaluation

### 1. Comprehensive Evaluation

Evaluate all models on test set:
```bash
python scripts/evaluate_models.py \
    --config configs/evaluation_config.yaml \
    --output_dir evaluation/reports
```

### 2. Individual Model Evaluation

**Detection Model:**
```bash
yolo detect val model=models/detection/best.pt data=data/processed/dataset.yaml
```

**Segmentation Model:**
```bash
yolo segment val model=models/segmentation/best.pt data=data/processed/dataset.yaml
```

### 3. Performance Targets

**Good Performance Indicators:**
- Detection mAP@0.5: >0.75
- Segmentation mIoU: >0.65
- Severity Accuracy: >85%
- Inference Time: <100ms per image

## Model Export and Deployment

### 1. Export Trained Models

Export models for production use:
```bash
# Export detection model
yolo export model=models/detection/best.pt format=onnx

# Export segmentation model  
yolo export model=models/segmentation/best.pt format=onnx

# Models will be saved with .onnx extension
```

### 2. Test Exported Models

Verify exported models work correctly:
```bash
python -c "
import onnxruntime as ort
import numpy as np

# Test ONNX model
session = ort.InferenceSession('models/detection/best.onnx')
input_shape = (1, 3, 640, 640)
dummy_input = np.random.randn(*input_shape).astype(np.float32)
output = session.run(None, {'images': dummy_input})
print('ONNX export successful!')
"
```

## Web Application Setup

### 1. Launch Streamlit App

Start the web interface:
```bash
streamlit run app/streamlit_app.py
```

Access the application at: http://localhost:8501

### 2. Test the Application

1. Upload a car damage image
2. Adjust confidence threshold
3. View detection and segmentation results
4. Check severity assessment
5. Export results in JSON/PNG format

## Training Schedule Recommendation

### Phase 1: Detection (Week 1)
- Day 1-2: Data preprocessing and verification
- Day 3-5: Detection model training and tuning
- Day 6-7: Detection model evaluation and optimization

### Phase 2: Segmentation (Week 2)  
- Day 1-3: Segmentation model training
- Day 4-5: Segmentation evaluation and fine-tuning
- Day 6-7: Joint detection+segmentation testing

### Phase 3: Severity Estimation (Week 3)
- Day 1-2: Severity model training
- Day 3-4: Severity evaluation and validation
- Day 5-7: End-to-end pipeline integration and testing

### Phase 4: Integration & Deployment (Week 4)
- Day 1-2: Complete pipeline evaluation
- Day 3-4: Web application development
- Day 5-7: Documentation, testing, and deployment

## Best Practices

### 1. Experiment Management
- Use descriptive experiment names
- Keep training logs and configurations
- Document hyperparameter changes
- Save model checkpoints regularly

### 2. Data Management
- Verify data integrity before training
- Use version control for datasets
- Monitor data distribution shifts
- Keep separate test sets untouched

### 3. Model Validation
- Use proper train/val/test splits
- Perform cross-validation for small datasets
- Test on diverse image conditions
- Validate on edge cases and failure modes

### 4. Performance Monitoring
- Track metrics consistently across experiments
- Use automated evaluation scripts
- Monitor inference speed and memory usage
- Test model robustness regularly

## Troubleshooting Guide

### Training Issues

**Problem: Model not learning (loss not decreasing)**
- Check learning rate (try 0.001, 0.01)
- Verify data loading (check dataset.yaml)
- Increase model complexity
- Check for data preprocessing issues

**Problem: Overfitting (validation loss increasing)**
- Add data augmentation
- Reduce model complexity
- Increase dropout
- Use early stopping

**Problem: Poor detection performance**
- Check anchor sizes and ratios
- Increase training epochs
- Adjust confidence thresholds
- Improve data quality

### Technical Issues

**Problem: CUDA out of memory**
- Reduce batch size
- Use gradient checkpointing
- Reduce image resolution
- Use mixed precision training

**Problem: Slow training**
- Use faster storage (SSD)
- Increase number of workers
- Use data caching
- Optimize data loading pipeline

## Support and Resources

### Documentation
- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Community
- GitHub Issues for bug reports
- Discussion forums for questions
- Technical documentation in `docs/` folder

### Contact
For technical support and questions, please refer to the project documentation or create an issue in the repository.

---

**Happy Training! 🚗🤖**
