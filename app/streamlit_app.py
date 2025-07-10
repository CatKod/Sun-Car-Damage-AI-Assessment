"""
Vehicle Damage Detection - Web Application
=========================================

This script creates a Streamlit web application for vehicle damage detection
with real-time inference, batch processing, and comprehensive analysis.

Features:
- Upload and analyze individual images
- Batch processing of multiple images
- Real-time damage detection and classification
- Detailed results visualization
- Model performance metrics
- Export results in various formats

Author: AI Assistant
Date: January 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import io
import base64
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tempfile
import zipfile
from datetime import datetime
import time
import glob
import warnings
warnings.filterwarnings('ignore')


class VehicleDamageApp:
    """Main application class for vehicle damage detection"""
    
    def __init__(self):
        """Initialize the application"""
        self.setup_page_config()
        self.load_models()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Vehicle Damage Detection AI",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ff7f0e;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .metric-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .damage-detected {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin: 1rem 0;
        }
        .no-damage {
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_models(self):
        """Load the trained YOLO models"""
        self.model = None
        self.model_path = None
        self.available_models = self.get_available_models()
        
        # Default class names for car damage detection
        self.default_class_names = [
            'scratches', 'dents', 'cracks', 'rust', 'missing_parts',
            'broken_lights', 'flat_tire', 'bumper_damage'
        ]
        self.class_names = self.default_class_names.copy()
        
        # Try to load the best available model
        if self.available_models:
            best_model = self.available_models[0]  # First one should be the best trained model
            try:
                self.model = YOLO(best_model)
                self.model_path = best_model
                
                # Update class names from model if available
                if hasattr(self.model, 'names') and self.model.names:
                    self.class_names = list(self.model.names.values())
                
                st.success(f"✅ Loaded model: {Path(best_model).name}")
                
            except Exception as e:
                st.error(f"❌ Failed to load model from {best_model}: {e}")
                self.try_fallback_model()
        else:
            st.warning("No trained models found. Trying fallback model...")
            self.try_fallback_model()
    
    def get_available_models(self):
        """Get list of available trained models"""
        models = []
        
        # Look for trained models in order of preference
        search_patterns = [
            "runs/*/weights/best.pt",  # Trained models
            "models/best.pt",          # Best model in models folder
            "models/yolo11n.pt",       # Base YOLO11n model
            "*.pt"                     # Any .pt file in root
        ]
        
        for pattern in search_patterns:
            found_models = glob.glob(pattern)
            for model_path in found_models:
                if Path(model_path).exists() and model_path not in models:
                    models.append(model_path)
        
        # Sort by modification time (newest first) for trained models
        trained_models = [m for m in models if 'runs/' in m]
        other_models = [m for m in models if 'runs/' not in m]
        
        if trained_models:
            trained_models.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        return trained_models + other_models
    
    def try_fallback_model(self):
        """Try to load a fallback model"""
        fallback_models = ['yolo11n.pt', 'yolov8n.pt']
        
        for model_name in fallback_models:
            try:
                st.info(f"Trying to load {model_name}...")
                self.model = YOLO(model_name)
                self.model_path = model_name
                
                # Update class names for pretrained model
                if hasattr(self.model, 'names') and self.model.names:
                    self.class_names = list(self.model.names.values())
                else:
                    self.class_names = self.default_class_names
                
                st.info(f"✅ Loaded fallback model: {model_name}")
                return
                
            except Exception as e:
                st.error(f"Failed to load {model_name}: {e}")
        
        st.error("❌ No models could be loaded. Please upload a model or check your installation.")
        self.model = None
        self.model_path = None
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = []
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<div class="main-header">🚗 Vehicle Damage Detection AI</div>', 
                   unsafe_allow_html=True)
        st.markdown("**Powered by YOLOv8 - Real-time Vehicle Damage Assessment**")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.title("⚙️ Configuration")
        
        # Model selection and status
        st.sidebar.subheader("🤖 Model Status")
        
        if self.model:
            model_name = Path(self.model_path).name if self.model_path else "Unknown"
            st.sidebar.success(f"✅ Active Model: {model_name}")
            
            # Show model details
            with st.sidebar.expander("📊 Model Details"):
                if self.model_path:
                    st.write(f"**Path:** {self.model_path}")
                    
                    # Show file size
                    try:
                        file_size = Path(self.model_path).stat().st_size / (1024*1024)  # MB
                        st.write(f"**Size:** {file_size:.1f} MB")
                    except:
                        pass
                
                st.write(f"**Classes:** {len(self.class_names)}")
                
                # Show class names
                if self.class_names:
                    st.write("**Damage Classes:**")
                    for i, class_name in enumerate(self.class_names[:10]):  # Show first 10
                        display_name = class_name.replace('_', ' ').title()
                        st.write(f"  {i}: {display_name}")
                    if len(self.class_names) > 10:
                        st.write(f"  ... and {len(self.class_names) - 10} more")
        else:
            st.sidebar.error("❌ No model loaded")
            if st.sidebar.button("🔄 Retry Model Loading"):
                self.load_models()
                st.rerun()
        
        # Model selection dropdown
        if self.available_models and len(self.available_models) > 1:
            st.sidebar.subheader("🔄 Switch Model")
            
            # Create display names for models
            model_options = {}
            for model_path in self.available_models:
                if 'runs/' in model_path:
                    # Extract run name for trained models
                    run_name = model_path.split('runs/')[1].split('/')[0]
                    display_name = f"📈 {run_name[:30]}..." if len(run_name) > 30 else f"📈 {run_name}"
                else:
                    display_name = f"🔧 {Path(model_path).name}"
                model_options[display_name] = model_path
            
            # Current model selection
            current_display = None
            for display, path in model_options.items():
                if path == self.model_path:
                    current_display = display
                    break
            
            selected_display = st.sidebar.selectbox(
                "Select Model:",
                list(model_options.keys()),
                index=list(model_options.keys()).index(current_display) if current_display else 0
            )
            
            selected_path = model_options[selected_display]
            if selected_path != self.model_path:
                if st.sidebar.button("🔄 Load Selected Model"):
                    self.switch_model(selected_path)
                    st.rerun()
        
        # Model upload section
        st.sidebar.subheader("📤 Upload Custom Model")
        uploaded_model = st.sidebar.file_uploader(
            "Upload YOLO Model (.pt)", 
            type=['pt'],
            help="Upload a trained YOLO model file (.pt format)"
        )
        if uploaded_model:
            self.load_uploaded_model(uploaded_model)
        
        # Detection parameters
        st.sidebar.subheader("🎯 Detection Parameters")
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25, 
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        iou_threshold = st.sidebar.slider(
            "IoU Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.45, 
            step=0.05,
            help="IoU threshold for Non-Maximum Suppression"
        )
        
        # Advanced settings
        with st.sidebar.expander("⚙️ Advanced Settings"):
            max_det = st.slider(
                "Max Detections", 
                min_value=10, 
                max_value=1000, 
                value=300,
                help="Maximum number of detections per image"
            )
            
            image_size = st.selectbox(
                "Image Size",
                [416, 640, 832, 1280],
                index=1,
                help="Input image size for inference"
            )
        
        # Model performance info (if available)
        if self.model and self.model_path and 'runs/' in self.model_path:
            model_dir = Path(self.model_path).parent.parent  # Go up to run directory
            results_file = model_dir / 'results.csv'
            
            if results_file.exists():
                with st.sidebar.expander("📈 Training Results"):
                    try:
                        results_df = pd.read_csv(results_file)
                        if not results_df.empty:
                            last_epoch = results_df.iloc[-1]
                            
                            # Show key metrics
                            metrics_to_show = [
                                ('mAP50', ['metrics/mAP50(B)', 'mAP50']),
                                ('mAP50-95', ['metrics/mAP50-95(B)', 'mAP50-95']),
                                ('Precision', ['metrics/precision(B)', 'precision']),
                                ('Recall', ['metrics/recall(B)', 'recall'])
                            ]
                            
                            for display_name, possible_cols in metrics_to_show:
                                value = None
                                for col_name in possible_cols:
                                    if col_name in last_epoch and pd.notna(last_epoch[col_name]):
                                        value = last_epoch[col_name]
                                        break
                                
                                if value is not None:
                                    st.metric(display_name, f"{value:.3f}")
                            
                            # Show training epochs
                            if 'epoch' in results_df.columns:
                                total_epochs = len(results_df)
                                st.write(f"**Epochs:** {total_epochs}")
                        
                    except Exception as e:
                        st.write(f"Error loading results: {e}")
        
        # Model benchmark section
        if self.model:
            st.sidebar.subheader("⚡ Model Benchmark")
            if st.sidebar.button("🚀 Run Speed Test"):
                with st.sidebar.spinner("Running benchmark..."):
                    benchmark_results = self.benchmark_model()
                
                if benchmark_results:
                    st.sidebar.success("Benchmark completed!")
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        st.metric("Avg Time", f"{benchmark_results['avg_time_ms']:.1f}ms")
                        st.metric("Min Time", f"{benchmark_results['min_time_ms']:.1f}ms")
                    with col2:
                        st.metric("Max Time", f"{benchmark_results['max_time_ms']:.1f}ms")
                        st.metric("FPS", f"{benchmark_results['fps']:.1f}")
                else:
                    st.sidebar.error("Benchmark failed")

        return conf_threshold, iou_threshold, max_det, image_size
    
    def switch_model(self, model_path):
        """Switch to a different model"""
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            
            # Update class names from the new model
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = self.default_class_names
            
            st.sidebar.success(f"✅ Switched to model: {Path(model_path).name}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load model {model_path}: {str(e)}")
    
    def load_uploaded_model(self, uploaded_file):
        """Load a model uploaded by the user"""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Try to load the model
            self.model = YOLO(temp_path)
            self.model_path = temp_path
            
            # Update class names
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = self.default_class_names
            
            st.sidebar.success(f"✅ Uploaded model loaded successfully!")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ Failed to load uploaded model: {str(e)}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and Path(temp_path).exists():
                Path(temp_path).unlink()
    
    def predict_image(self, image, conf_threshold=0.25, iou_threshold=0.45, max_det=300, imgsz=640):
        """Run inference on an image"""
        if not self.model:
            return None
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Run inference
            results = self.model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                imgsz=imgsz,
                verbose=False
            )
            
            return results[0] if results else None
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def draw_detections(self, image, results):
        """Draw detection boxes and labels on image"""
        if not results or not hasattr(results, 'boxes') or results.boxes is None:
            return image
        
        # Convert to PIL Image for drawing
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        draw = ImageDraw.Draw(image)
        
        # Colors for different classes
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
            '#00FFFF', '#FFA500', '#800080', '#008000', '#FFC0CB'
        ]
        
        boxes = results.boxes
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()
            conf = boxes.conf[i].item()
            cls = int(boxes.cls[i].item())
            
            # Get class name
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class {cls}"
            
            # Choose color
            color = colors[cls % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            
            # Use default font
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Get text size and draw background
            if font:
                bbox = draw.textbbox((x1, y1-25), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1-25), label, fill='white', font=font)
            else:
                draw.text((x1, y1-25), label, fill=color)
        
        return image
    
    def load_uploaded_model(self, uploaded_model):
        """Load model from uploaded file"""
        try:
            # Save uploaded model temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_model.read())
                tmp_path = tmp_file.name
            
            # Load model
            self.model = YOLO(tmp_path)
            self.model_path = tmp_path
            
            # Update class names if available
            if hasattr(self.model, 'names') and self.model.names:
                # Check if it's likely a damage detection model
                if any(name in uploaded_model.name.lower() for name in ['damage', 'vehicle', 'car']):
                    # Keep our custom damage classes
                    pass
                else:
                    # Use model's classes
                    self.class_names = list(self.model.names.values())
            
            st.sidebar.success("✅ Model loaded successfully!")
            st.experimental_rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
    
    def predict_image(self, image, conf_threshold=0.25, iou_threshold=0.45, max_det=300, imgsz=640):
        """Make prediction on a single image"""
        if not self.model:
            return None, None, "No model loaded"
        
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Make prediction with enhanced parameters
            results = self.model.predict(
                img_array, 
                conf=conf_threshold, 
                iou=iou_threshold,
                max_det=max_det,
                imgsz=imgsz,
                verbose=False
            )
            
            if not results or len(results) == 0:
                return None, None, "No results returned"
            
            result = results[0]
            
            # Extract detections
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Ensure class_id is within bounds
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox.tolist(),
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }
                    detections.append(detection)
            
            # Get annotated image
            annotated_img = result.plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            return detections, annotated_img, None
            
        except Exception as e:
            return None, None, str(e)
    
    def render_single_image_analysis(self, conf_threshold, iou_threshold, max_det, imgsz):
        """Render single image analysis interface"""
        st.markdown('<div class="sub-header">📸 Single Image Analysis</div>', 
                   unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Vehicle Image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a vehicle for damage detection"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image",use_container_width=True)
            
            # Make prediction
            with st.spinner("Analyzing image for damage..."):
                detections, annotated_img, error = self.predict_image(
                    image, conf_threshold, iou_threshold, max_det, imgsz
                )
            
            if error:
                st.error(f"Error during prediction: {error}")
                return
            
            with col2:
                st.subheader("Detection Results")
                if annotated_img is not None:
                    st.image(annotated_img, caption="Detected Damage",use_container_width=True)
                else:
                    st.image(image, caption="No damage detected",use_container_width=True)
            
            # Display results
            self.display_detection_results(detections, uploaded_file.name)
            
            # Save to history
            if detections is not None:
                result_data = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': uploaded_file.name,
                    'detections': detections,
                    'num_detections': len(detections)
                }
                st.session_state.analysis_history.append(result_data)
    
    def display_detection_results(self, detections, filename):
        """Display detection results in a formatted way"""
        if not detections:
            st.markdown(
                '<div class="no-damage">✅ <b>No damage detected</b> - Vehicle appears to be in good condition</div>',
                unsafe_allow_html=True
            )
            return
        
        # Damage detected
        st.markdown(
            f'<div class="damage-detected">⚠️ <b>{len(detections)} damage(s) detected</b></div>',
            unsafe_allow_html=True
        )
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", len(detections))
        
        with col2:
            avg_confidence = np.mean([d['confidence'] for d in detections])
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            unique_classes = len(set([d['class_name'] for d in detections]))
            st.metric("Damage Types", unique_classes)
        
        with col4:
            max_confidence = max([d['confidence'] for d in detections])
            st.metric("Max Confidence", f"{max_confidence:.3f}")
        
        # Detailed results table
        st.subheader("Detailed Detection Results")
        
        results_df = pd.DataFrame([
            {
                'Damage Type': d['class_name'].replace('_', ' ').title(),
                'Confidence': f"{d['confidence']:.3f}",
                'Area (pixels)': f"{d['area']:.0f}",
                'Severity': self.estimate_severity(d['confidence'], d['area'])
            }
            for d in detections
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Damage distribution chart
        if len(detections) > 1:
            self.plot_damage_distribution(detections)
    
    def estimate_severity(self, confidence, area):
        """Estimate damage severity based on confidence and area"""
        if confidence > 0.8 and area > 10000:
            return "High"
        elif confidence > 0.6 and area > 5000:
            return "Medium"
        else:
            return "Low"
    
    def plot_damage_distribution(self, detections):
        """Plot damage type distribution"""
        damage_counts = {}
        for detection in detections:
            damage_type = detection['class_name'].replace('_', ' ').title()
            damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
        
        fig = px.pie(
            values=list(damage_counts.values()),
            names=list(damage_counts.keys()),
            title="Damage Type Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_batch_analysis(self, conf_threshold, iou_threshold, max_det, imgsz):
        """Render batch analysis interface"""
        st.markdown('<div class="sub-header">📁 Batch Analysis</div>', 
                   unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload Multiple Vehicle Images", 
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            if st.button("🔍 Analyze Batch", type="primary"):
                self.process_batch(uploaded_files, conf_threshold, iou_threshold, max_det, imgsz)
        
        # Display batch results
        if st.session_state.batch_results:
            self.display_batch_results()
    
    def process_batch(self, uploaded_files, conf_threshold, iou_threshold, max_det, imgsz):
        """Process multiple images in batch"""
        batch_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Load image
            image = Image.open(uploaded_file)
            
            # Make prediction
            detections, annotated_img, error = self.predict_image(
                image, conf_threshold, iou_threshold, max_det, imgsz
            )
            
            result = {
                'filename': uploaded_file.name,
                'detections': detections or [],
                'num_detections': len(detections) if detections else 0,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add damage summary
            if detections:
                damage_types = [d['class_name'] for d in detections]
                result['damage_types'] = list(set(damage_types))
                result['avg_confidence'] = np.mean([d['confidence'] for d in detections])
                result['max_confidence'] = max([d['confidence'] for d in detections])
            else:
                result['damage_types'] = []
                result['avg_confidence'] = 0.0
                result['max_confidence'] = 0.0
            
            batch_results.append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.session_state.batch_results = batch_results
        status_text.text("Batch processing complete!")
        st.success(f"✅ Successfully processed {len(uploaded_files)} images")
    
    def display_batch_results(self):
        """Display batch processing results"""
        results = st.session_state.batch_results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_images = len(results)
        damaged_images = sum(1 for r in results if r['num_detections'] > 0)
        total_detections = sum(r['num_detections'] for r in results)
        avg_detections = total_detections / total_images if total_images > 0 else 0
        
        with col1:
            st.metric("Total Images", total_images)
        with col2:
            st.metric("Damaged Images", damaged_images)
        with col3:
            st.metric("Total Detections", total_detections)
        with col4:
            st.metric("Avg Detections/Image", f"{avg_detections:.1f}")
        
        # Results table
        st.subheader("Batch Results Summary")
        
        results_df = pd.DataFrame([
            {
                'Filename': r['filename'],
                'Detections': r['num_detections'],
                'Damage Types': ', '.join([dt.replace('_', ' ').title() for dt in r['damage_types']]),
                'Max Confidence': f"{r['max_confidence']:.3f}" if r['max_confidence'] > 0 else "N/A",
                'Status': "✅ Clean" if r['num_detections'] == 0 else f"⚠️ {r['num_detections']} damage(s)"
            }
            for r in results
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Batch statistics
        self.plot_batch_statistics(results)
        
        # Export options
        self.render_export_options(results)
    
    def plot_batch_statistics(self, results):
        """Plot batch processing statistics"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Detection count distribution
            detection_counts = [r['num_detections'] for r in results]
            fig1 = px.histogram(
                x=detection_counts,
                nbins=max(1, max(detection_counts) if detection_counts else 1),
                title="Distribution of Detections per Image"
            )
            fig1.update_layout(xaxis_title="Number of Detections", yaxis_title="Number of Images")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Damage type distribution
            all_damage_types = []
            for r in results:
                all_damage_types.extend(r['damage_types'])
            
            if all_damage_types:
                damage_counts = {}
                for damage_type in all_damage_types:
                    clean_name = damage_type.replace('_', ' ').title()
                    damage_counts[clean_name] = damage_counts.get(clean_name, 0) + 1
                
                fig2 = px.bar(
                    x=list(damage_counts.keys()),
                    y=list(damage_counts.values()),
                    title="Damage Type Distribution"
                )
                fig2.update_layout(xaxis_title="Damage Type", yaxis_title="Count")
                st.plotly_chart(fig2, use_container_width=True)
    
    def render_export_options(self, results):
        """Render export options for batch results"""
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to CSV"):
                csv_data = self.export_to_csv(results)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"vehicle_damage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Export to JSON"):
                json_data = self.export_to_json(results)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"vehicle_damage_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("📈 Export Report"):
                report_data = self.export_report(results)
                st.download_button(
                    label="Download Report",
                    data=report_data,
                    file_name=f"vehicle_damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    def export_to_csv(self, results):
        """Export results to CSV format"""
        data = []
        for r in results:
            if r['detections']:
                for detection in r['detections']:
                    data.append({
                        'filename': r['filename'],
                        'damage_type': detection['class_name'],
                        'confidence': detection['confidence'],
                        'area': detection['area'],
                        'severity': self.estimate_severity(detection['confidence'], detection['area'])
                    })
            else:
                data.append({
                    'filename': r['filename'],
                    'damage_type': 'no_damage',
                    'confidence': 0.0,
                    'area': 0.0,
                    'severity': 'None'
                })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def export_to_json(self, results):
        """Export results to JSON format"""
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_images': len(results),
            'model_path': self.model_path,
            'results': results
        }
        return json.dumps(export_data, indent=2)
    
    def export_report(self, results):
        """Export comprehensive report"""
        total_images = len(results)
        damaged_images = sum(1 for r in results if r['num_detections'] > 0)
        total_detections = sum(r['num_detections'] for r in results)
        
        report = f"""# Vehicle Damage Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model:** {self.model_path or 'Unknown'}

## Summary Statistics

- **Total Images Analyzed:** {total_images}
- **Images with Damage:** {damaged_images} ({damaged_images/total_images*100:.1f}%)
- **Images without Damage:** {total_images - damaged_images} ({(total_images - damaged_images)/total_images*100:.1f}%)
- **Total Detections:** {total_detections}
- **Average Detections per Image:** {total_detections/total_images:.2f}

## Damage Distribution

"""
        
        # Add damage type distribution
        all_damage_types = []
        for r in results:
            all_damage_types.extend(r['damage_types'])
        
        if all_damage_types:
            damage_counts = {}
            for damage_type in all_damage_types:
                clean_name = damage_type.replace('_', ' ').title()
                damage_counts[clean_name] = damage_counts.get(clean_name, 0) + 1
            
            for damage_type, count in damage_counts.items():
                report += f"- **{damage_type}:** {count} occurrences\n"
        
        report += "\n## Detailed Results\n\n"
        
        for r in results:
            report += f"### {r['filename']}\n"
            if r['num_detections'] > 0:
                report += f"- **Status:** ⚠️ {r['num_detections']} damage(s) detected\n"
                report += f"- **Damage Types:** {', '.join([dt.replace('_', ' ').title() for dt in r['damage_types']])}\n"
                report += f"- **Max Confidence:** {r['max_confidence']:.3f}\n"
            else:
                report += "- **Status:** ✅ No damage detected\n"
            report += "\n"
        
        return report
    
    def render_analysis_history(self):
        """Render analysis history"""
        st.markdown('<div class="sub-header">📊 Analysis History</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.analysis_history:
            st.info("No analysis history available. Upload and analyze images to see history.")
            return
        
        # Display history
        history_df = pd.DataFrame([
            {
                'Timestamp': h['timestamp'],
                'Filename': h['filename'],
                'Detections': h['num_detections'],
                'Status': "✅ Clean" if h['num_detections'] == 0 else f"⚠️ {h['num_detections']} damage(s)"
            }
            for h in st.session_state.analysis_history
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.experimental_rerun()
    
    def get_model_info(self):
        """Get detailed model information"""
        if not self.model:
            return None
        
        try:
            info = {
                'model_path': self.model_path,
                'model_name': Path(self.model_path).name if self.model_path else "Unknown",
                'num_classes': len(self.class_names),
                'classes': self.class_names
            }
            
            # Try to get additional model info
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'yaml'):
                    info['model_type'] = self.model.model.yaml.get('backbone', 'Unknown')
                
                # Get model size info
                try:
                    total_params = sum(p.numel() for p in self.model.model.parameters())
                    info['total_parameters'] = f"{total_params:,}"
                except:
                    info['total_parameters'] = "Unknown"
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def benchmark_model(self, num_runs=5):
        """Benchmark model inference speed"""
        if not self.model:
            return None
        
        import time
        import numpy as np
        
        # Create a dummy image for benchmarking
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            try:
                results = self.model.predict(dummy_image, verbose=False)
                end_time = time.time()
                times.append(end_time - start_time)
            except:
                return None
        
        if times:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            return {
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'fps': fps
            }
        
        return None
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Get configuration from sidebar
        conf_threshold, iou_threshold, max_det, imgsz = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Analysis", "📊 History"])
        
        with tab1:
            self.render_single_image_analysis(conf_threshold, iou_threshold, max_det, imgsz)
        
        with tab2:
            self.render_batch_analysis(conf_threshold, iou_threshold, max_det, imgsz)
        
        with tab3:
            self.render_analysis_history()
        
        # Footer
        st.markdown("---")
        st.markdown("**Vehicle Damage Detection AI** - Powered by YOLOv8 & Streamlit")


def main():
    """Main execution function"""
    app = VehicleDamageApp()
    app.run()


if __name__ == "__main__":
    main()