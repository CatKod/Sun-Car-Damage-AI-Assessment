# Car Damage Analysis - yolo11n Training Report

## Training Configuration
- **Model**: models/yolo11n.pt
- **Dataset**: yolo_dataset
- **Device**: cuda
- **Batch Size**: 4
- **Training Date**: 2025-07-10T00:58:04.563413

## Detection Performance

- **Validation mAP@0.5**: 0.7249
- **Validation mAP@0.5:0.95**: 0.5761
- **Test mAP@0.5**: 0.7357
- **Test mAP@0.5:0.95**: 0.5833
- **Test Precision**: 0.7353
- **Test Recall**: 0.6878
- **Test F1**: 0.7108

## Robustness Analysis
- **Overall Stability**: moderate
- **Angle Variance**: 0.0064
- **Completeness Variance**: 0.3025

## Model Output
- **Detection Model**: `runs\car_damage_yolo11n_20250709_135429\weights\best.pt`
