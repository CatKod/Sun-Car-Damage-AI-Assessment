# Vehicle Damage Detection - Streamlit Web Application

## Overview

The rebuilt Streamlit application provides a comprehensive interface for vehicle damage detection using YOLO models. The app automatically discovers and loads trained models from your project.

## Features

### 🚗 **Automatic Model Discovery**
- Automatically finds trained models in `runs/*/weights/best.pt`
- Discovers base models in `models/` directory
- Supports fallback to pretrained YOLO models
- Easy model switching through dropdown interface

### 📸 **Single Image Analysis**
- Upload individual vehicle images (PNG, JPG, JPEG)
- Real-time damage detection and classification
- Detailed confidence scores and bounding boxes
- Visual damage type distribution charts

### 📁 **Batch Processing**
- Upload multiple images for batch analysis
- Progress tracking for large batches
- Comprehensive summary statistics
- Export results in JSON format

### ⚙️ **Configurable Settings**
- Adjustable confidence threshold (0.0 - 1.0)
- IoU threshold for Non-Maximum Suppression
- Maximum detections per image
- Input image size selection

### 📊 **Advanced Analytics**
- Training metrics display (for trained models)
- Model performance benchmarking
- Analysis history tracking
- Detailed detection summaries

## Quick Start

### 1. Run the Application

**Method 1: Using provided scripts**
```bash
# Windows
run_app.bat

# Linux/Mac
chmod +x run_app.sh
./run_app.sh
```

**Method 2: Direct command**
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

### 2. Access the Interface

Open your browser and navigate to: http://localhost:8501

### 3. Using the Application

#### **Model Management**
1. **Automatic Loading**: The app loads the best available model automatically
2. **Model Switching**: Use the sidebar dropdown to switch between available models
3. **Custom Upload**: Upload your own YOLO model files (.pt format)
4. **Model Details**: View model information, class names, and training metrics

#### **Single Image Analysis**
1. Navigate to the "📸 Single Image" tab
2. Upload a vehicle image using the file uploader
3. Click "🔍 Analyze Image" to run detection
4. View results with annotated image and detailed statistics

#### **Batch Processing**
1. Navigate to the "📁 Batch Analysis" tab
2. Upload multiple images using the multi-file uploader
3. Click "🔍 Analyze Batch" to process all images
4. Review summary statistics and download results

#### **History Tracking**
1. Navigate to the "📊 History" tab
2. View all previous analyses
3. Clear history if needed

### 4. Configuration Options

#### **Detection Parameters**
- **Confidence Threshold**: Minimum confidence score for detections (default: 0.25)
- **IoU Threshold**: Intersection over Union threshold for NMS (default: 0.45)

#### **Advanced Settings**
- **Max Detections**: Maximum number of detections per image (default: 300)
- **Image Size**: Input resolution for inference (416, 640, 832, 1280)

## Model Requirements

### **Compatible Model Formats**
- YOLO model files (.pt format)
- Trained on vehicle damage detection tasks
- Compatible with Ultralytics YOLO library

### **Expected Model Locations**
1. `runs/*/weights/best.pt` - Trained models (automatically prioritized)
2. `models/best.pt` - Best model in models directory
3. `models/yolo11n.pt` - Base YOLO11n model
4. `*.pt` - Any .pt file in project root

### **Class Names**
The app automatically detects class names from the model. For custom models without embedded class names, it falls back to default damage categories:
- Scratches
- Dents
- Cracks
- Rust
- Missing Parts
- Broken Lights
- Flat Tire
- Bumper Damage

## Troubleshooting

### **Common Issues**

#### **No Model Loaded**
- **Cause**: No compatible models found
- **Solution**: 
  1. Train a model using the project scripts
  2. Upload a custom model through the sidebar
  3. Ensure model files are in expected locations

#### **Import Errors**
- **Cause**: Missing dependencies
- **Solution**: Install requirements
  ```bash
  pip install -r app/requirements.txt
  ```

#### **Performance Issues**
- **Cause**: Large images or high detection settings
- **Solution**: 
  1. Reduce image size in advanced settings
  2. Lower max detections limit
  3. Increase confidence threshold

#### **Model Loading Errors**
- **Cause**: Corrupted or incompatible model files
- **Solution**: 
  1. Re-download or retrain the model
  2. Check model file integrity
  3. Ensure YOLO compatibility

### **Performance Tips**

1. **Optimize Settings**: Use appropriate confidence and IoU thresholds
2. **Image Size**: Use 640px for balance of speed and accuracy
3. **Batch Size**: Process images in smaller batches for better responsiveness
4. **Model Selection**: Choose models appropriate for your use case

## Technical Details

### **Architecture**
- **Frontend**: Streamlit web framework
- **Backend**: Ultralytics YOLO for inference
- **Visualization**: Plotly for interactive charts
- **Image Processing**: OpenCV and PIL for image handling

### **File Structure**
```
app/
├── streamlit_app.py      # Main application file
├── requirements.txt      # Package dependencies
└── streamlit_app_old.py  # Backup of previous version
```

### **Dependencies**
- streamlit>=1.28.0
- ultralytics>=8.0.0
- opencv-python>=4.8.0
- pillow>=9.0.0
- numpy>=1.21.0
- pandas>=1.3.0
- plotly>=5.0.0
- matplotlib>=3.5.0

## Support

If you encounter issues:
1. Check the console for error messages
2. Verify all dependencies are installed
3. Ensure model files are accessible
4. Try refreshing the browser page
5. Restart the Streamlit server
