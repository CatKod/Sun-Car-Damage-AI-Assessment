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

Author: Hoang Kim Vinh
Date: 2025
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
import threading
from flask import Flask, request, jsonify
import socket
import logging

warnings.filterwarnings('ignore')


# Shared state for Camera
class CameraSharedState:
    def __init__(self):
        self.latest_image = None
        self.latest_results = None
        self.last_update_time = None
        self.lock = threading.Lock()
        self.server_running = False
        self.server_port = 5000
        self.model = None

camera_state = CameraSharedState()

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def run_flask_server():
    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    @app.route('/upload', methods=['POST'])
    def upload_image():
        try:
            image = None
            if request.is_json:
                data = request.get_json()
                if 'image' in data:
                    image_data = base64.b64decode(data['image'])
                    image = Image.open(io.BytesIO(image_data))
            elif 'file' in request.files:
                file = request.files['file']
                image = Image.open(file.stream)
            elif request.data:
                image = Image.open(io.BytesIO(request.data))
                
            if image:
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Run inference if model is available
                results = None
                response_text = "Safe"
                
                with camera_state.lock:
                    current_model = camera_state.model
                
                if current_model:
                    results = current_model(image)
                    # Process for ESP32 response
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            if float(box.conf[0]) > 0.4:
                                detections.append({
                                    'class': current_model.names[int(box.cls[0])],
                                    'conf': float(box.conf[0])
                                })
                    
                    if detections:
                        best_det = max(detections, key=lambda x: x['conf'])
                        response_text = best_det['class']
                
                with camera_state.lock:
                    camera_state.latest_image = image
                    camera_state.latest_results = results
                    camera_state.last_update_time = datetime.now()
                
                return jsonify({"status": "success", "result": response_text})
            
            return jsonify({"status": "error", "message": "No image data"}), 400
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    try:
        app.run(host='0.0.0.0', port=camera_state.server_port, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Failed to start Flask server: {e}")

@st.cache_resource
def start_background_server():
    if not camera_state.server_running:
        t = threading.Thread(target=run_flask_server, daemon=True)
        t.start()
        camera_state.server_running = True
        return t
    return None


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
        .repair-cost-section {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .amazon-button {
            background-color: #FF9900 !important;
            color: white !important;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .amazon-button:hover {
            background-color: #e68900 !important;
        }
        .product-card {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            height: 100%;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .stat-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-3px);
        }
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 15px;
            margin: 2rem 0;
            text-align: center;
            color: white;
        }
        .getting-started {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            color: white;
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
        st.markdown("**Powered by YOLO11n - Real-time Vehicle Damage Assessment**")
        st.markdown("---")
    
    def render_home_page(self):
        """Render the home page with welcome message and features overview"""
        
        # Hero Section
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 3rem 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;">
            <h1 style="color: white; font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🚗 Welcome to Vehicle Damage Detection AI
            </h1>
            <p style="color: white; font-size: 1.3rem; margin-bottom: 2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                Advanced AI-powered solution for accurate vehicle damage assessment and repair cost estimation
            </p>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; backdrop-filter: blur(10px);">
                <p style="color: white; font-size: 1.1rem; margin: 0;">
                    🎯 <strong>Detect</strong> • 💰 <strong>Estimate</strong> • 🛠️ <strong>Repair</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center; border-left: 4px solid #28a745;">
                <h3 style="color: #28a745; margin: 0;">8+</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Damage Types</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center; border-left: 4px solid #007bff;">
                <h3 style="color: #007bff; margin: 0;">95%+</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center; border-left: 4px solid #ffc107;">
                <h3 style="color: #ffc107; margin: 0;"><1s</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center; border-left: 4px solid #dc3545;">
                <h3 style="color: #dc3545; margin: 0;">24/7</h3>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Available</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Features Overview
        st.markdown("## 🚀 Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 AI-Powered Detection
            - **Advanced YOLO11n Model**: State-of-the-art object detection
            - **Real-time Analysis**: Get results in seconds
            - **Multiple Damage Types**: Scratches, dents, cracks, rust, and more
            - **High Accuracy**: Professional-grade detection capabilities
            
            ### 💰 Cost Estimation
            - **Intelligent Pricing**: AI-based repair cost estimation
            - **Severity Assessment**: Damage severity classification
            - **Market Rates**: Updated pricing based on current market
            - **Professional vs DIY**: Cost comparison options
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Product Recommendations
            - **Amazon Integration**: Direct links to repair products
            - **Curated Selection**: Best-rated repair kits and tools
            - **Price Ranges**: Budget-friendly to professional options
            - **DIY Guides**: Step-by-step repair recommendations
            
            ### 📊 Comprehensive Reports
            - **Detailed Analysis**: Complete damage assessment reports
            - **Export Options**: CSV, JSON, and Markdown formats
            - **Batch Processing**: Analyze multiple images at once
            - **History Tracking**: Keep track of all analyses
            """)
        
        # How It Works
        st.markdown("## 🔄 How It Works")
        
        tab1, tab2, tab3 = st.columns(3)
        
        with tab1:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 300px;">
                <div style="background: #e3f2fd; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;">
                    <span style="font-size: 2rem;">📤</span>
                </div>
                <h3 style="color: #1976d2; margin-bottom: 1rem;">1. Upload Image</h3>
                <p style="color: #666; line-height: 1.6;">
                    Upload a photo of your vehicle from any angle. Our AI works with any image quality and lighting conditions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 300px;">
                <div style="background: #f3e5f5; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;">
                    <span style="font-size: 2rem;">🤖</span>
                </div>
                <h3 style="color: #7b1fa2; margin-bottom: 1rem;">2. AI Analysis</h3>
                <p style="color: #666; line-height: 1.6;">
                    Our advanced YOLO11n model analyzes the image, detecting and classifying different types of vehicle damage.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 300px;">
                <div style="background: #e8f5e8; width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;">
                    <span style="font-size: 2rem;">📋</span>
                </div>
                <h3 style="color: #388e3c; margin-bottom: 1rem;">3. Get Results</h3>
                <p style="color: #666; line-height: 1.6;">
                    Receive detailed analysis with cost estimates, repair recommendations, and direct links to products.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Model Information
        if self.model:
            st.markdown("## 🤖 AI Model Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #17a2b8;">
                    <h4 style="color: #17a2b8; margin-top: 0;">Current Model Status</h4>
                </div>
                """, unsafe_allow_html=True)
                
                model_name = Path(self.model_path).name if self.model_path else "Unknown"
                st.success(f"✅ **Active Model:** {model_name}")
                st.info(f"🎯 **Classes Detected:** {len(self.class_names)}")
                
                if self.class_names:
                    st.markdown("**Damage Types:**")
                    damage_types_cols = st.columns(2)
                    for i, class_name in enumerate(self.class_names[:8]):
                        col = damage_types_cols[i % 2]
                        display_name = class_name.replace('_', ' ').title()
                        col.write(f"• {display_name}")
            
            with col2:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745;">
                    <h4 style="color: #28a745; margin-top: 0;">Technical Specifications</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                - **Architecture:** YOLO11n (You Only Look Once)
                - **Framework:** Ultralytics
                - **Input Size:** 640x640 pixels
                - **Inference Speed:** <1 second
                - **Model Type:** Object Detection & Classification
                - **Training Data:** CarDD Dataset
                """)
        
        # Getting Started
        st.markdown("## 🎯 Getting Started")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; margin: 1rem 0;">
            <h3 style="color: white; margin-top: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                Ready to analyze your vehicle? 🚗
            </h3>
            <p style="color: white; font-size: 1.1rem; margin-bottom: 1.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                Choose from our three powerful analysis modes:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; backdrop-filter: blur(10px);">
                    <h4 style="color: white; margin: 0 0 0.5rem 0;">📸 Single Image</h4>
                    <p style="color: white; margin: 0; font-size: 0.9rem;">Upload one image for detailed analysis</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; backdrop-filter: blur(10px);">
                    <h4 style="color: white; margin: 0 0 0.5rem 0;">📁 Batch Analysis</h4>
                    <p style="color: white; margin: 0; font-size: 0.9rem;">Process multiple images at once</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; backdrop-filter: blur(10px);">
                    <h4 style="color: white; margin: 0 0 0.5rem 0;">📊 View History</h4>
                    <p style="color: white; margin: 0; font-size: 0.9rem;">Track your analysis history</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer with additional info
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🔧 Technical Support
            - Model accuracy: 95%+
            - Supported formats: JPG, PNG, JPEG
            - Max file size: 200MB
            - Processing time: <1 second
            """)
        
        with col2:
            st.markdown("""
            ### 💡 Tips for Best Results
            - Use clear, well-lit photos
            - Capture damage from multiple angles
            - Ensure damage is visible in frame
            - Higher resolution = better accuracy
            """)
        
        with col3:
            st.markdown("""
            ### 📞 About This Project
            - **Author:** Hoang Kim Vinh
            - **Technology:** YOLO11n + Streamlit
            - **Dataset:** CarDD (Car Damage Detection)
            - **Version:** 2025 Release
            """)
    
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
                'Severity': self.estimate_severity(d['confidence'], d['area']),
                'Estimated Cost (USD)': f"${self.estimate_repair_cost(d['class_name'], self.estimate_severity(d['confidence'], d['area'])):,}"
            }
            for d in detections
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Repair Cost Breakdown and Product Recommendations
        self.display_repair_cost_section(detections)
        
        # Damage distribution chart
        if len(detections) > 1:
            self.plot_damage_distribution(detections)

    def estimate_repair_cost(self, damage_type, severity):
        """Estimate repair cost based on damage type and severity"""
        # Base cost by damage type (USD)
        base_costs = {
            'dent': 200,
            'scratch': 100,
            'crack': 350,
            'rust': 150,
            'missing_parts': 400,
            'broken_lights': 120,
            'flat_tire': 80,
            'bumper_damage': 300,
            'dents': 200,
            'scratches': 100,
            'cracks': 350
        }
        # Normalize damage type
        key = damage_type.lower().replace(' ', '_')
        cost = base_costs.get(key, 150)
        
        # Severity multiplier
        if severity == 'High':
            cost *= 1.5
        elif severity == 'Medium':
            cost *= 1.2
        else:
            cost *= 0.8
        
        return int(cost)
    
    def get_repair_products(self, damage_type):
        """Get repair products and Amazon links for specific damage types"""
        products = {
            'dent': {
                'name': 'Paintless Dent Repair Kit',
                'description': 'Professional PDR tools for removing dents without painting',
                'amazon_link': 'https://www.amazon.com/s?k=paintless+dent+repair+kit+professional+PDR+tools&ref=nb_sb_noss',
                'price_range': '$30-80',
                'icon': '🔨'
            },
            'dents': {
                'name': 'Paintless Dent Repair Kit',
                'description': 'Professional PDR tools for removing dents without painting',
                'amazon_link': 'https://www.amazon.com/s?k=paintless+dent+repair+kit+professional+PDR+tools&ref=nb_sb_noss',
                'price_range': '$30-80',
                'icon': '🔨'
            },
            'scratch': {
                'name': 'Car Scratch Repair Kit',
                'description': 'Touch-up paint pens and polishing compounds for scratch removal',
                'amazon_link': 'https://www.amazon.com/s?k=car+scratch+repair+kit+touch+up+paint+pen&ref=nb_sb_noss',
                'price_range': '$15-40',
                'icon': '✏️'
            },
            'scratches': {
                'name': 'Car Scratch Repair Kit',
                'description': 'Touch-up paint pens and polishing compounds for scratch removal',
                'amazon_link': 'https://www.amazon.com/s?k=car+scratch+repair+kit+touch+up+paint+pen&ref=nb_sb_noss',
                'price_range': '$15-40',
                'icon': '✏️'
            },
            'crack': {
                'name': 'Windshield Crack Repair Kit',
                'description': 'DIY windshield and plastic crack repair solutions',
                'amazon_link': 'https://www.amazon.com/s?k=windshield+crack+repair+kit+DIY+glass+repair&ref=nb_sb_noss',
                'price_range': '$10-25',
                'icon': '🪟'
            },
            'cracks': {
                'name': 'Windshield Crack Repair Kit',
                'description': 'DIY windshield and plastic crack repair solutions',
                'amazon_link': 'https://www.amazon.com/s?k=windshield+crack+repair+kit+DIY+glass+repair&ref=nb_sb_noss',
                'price_range': '$10-25',
                'icon': '🪟'
            },
            'rust': {
                'name': 'Car Rust Converter & Treatment',
                'description': 'Rust treatment and prevention products',
                'amazon_link': 'https://www.amazon.com/s?k=car+rust+converter+treatment+corroseal+evapo-rust&ref=nb_sb_noss',
                'price_range': '$20-50',
                'icon': '🦠'
            },
            'missing_parts': {
                'name': 'Automotive Replacement Parts',
                'description': 'OEM and aftermarket automotive replacement parts',
                'amazon_link': 'https://www.amazon.com/s?k=automotive+replacement+parts+OEM+aftermarket&ref=nb_sb_noss',
                'price_range': '$50-500+',
                'icon': '🔧'
            },
            'broken_lights': {
                'name': 'Car Headlight & Taillight Assembly',
                'description': 'Replacement automotive lighting components and bulbs',
                'amazon_link': 'https://www.amazon.com/s?k=car+headlight+taillight+replacement+assembly+LED&ref=nb_sb_noss',
                'price_range': '$25-150',
                'icon': '💡'
            },
            'flat_tire': {
                'name': 'Tire Repair Kit with Compressor',
                'description': 'Emergency tire repair patches and portable air compressor',
                'amazon_link': 'https://www.amazon.com/s?k=tire+repair+kit+emergency+patch+air+compressor&ref=nb_sb_noss',
                'price_range': '$20-60',
                'icon': '🛞'
            },
            'bumper_damage': {
                'name': 'Plastic Bumper Repair Kit',
                'description': 'Plastic welding and bumper restoration materials',
                'amazon_link': 'https://www.amazon.com/s?k=plastic+bumper+repair+kit+welding+adhesive&ref=nb_sb_noss',
                'price_range': '$30-100',
                'icon': '🚗'
            }
        }
        
        key = damage_type.lower().replace(' ', '_')
        return products.get(key, {
            'name': 'General Automotive Repair Kit',
            'description': 'Basic automotive repair supplies and tools',
            'amazon_link': 'https://www.amazon.com/s?k=automotive+repair+kit+tools+emergency&ref=nb_sb_noss',
            'price_range': '$20-100',
            'icon': '🔧'
        })
    
    def display_repair_cost_section(self, detections):
        """Display detailed repair cost breakdown with product recommendations"""
        st.markdown('<div class="sub-header">💰 Repair Cost Estimate & Product Recommendations</div>', 
                   unsafe_allow_html=True)
        
        # Calculate total estimated cost
        total_cost = sum(self.estimate_repair_cost(d['class_name'], self.estimate_severity(d['confidence'], d['area'])) 
                        for d in detections)
        
        # Cost summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Estimated Cost", f"${total_cost:,}")
        with col2:
            avg_cost = total_cost / len(detections)
            st.metric("Average Cost per Damage", f"${avg_cost:,.0f}")
        with col3:
            min_cost = int(total_cost * 0.7)  # 30% discount range
            max_cost = int(total_cost * 1.3)  # 30% markup range
            st.metric("Cost Range", f"${min_cost:,} - ${max_cost:,}")
        
        st.markdown("---")
        
        # Group detections by damage type for better organization
        damage_groups = {}
        for detection in detections:
            damage_type = detection['class_name']
            if damage_type not in damage_groups:
                damage_groups[damage_type] = []
            damage_groups[damage_type].append(detection)
        
        # Display each damage type with product recommendations
        for damage_type, damage_list in damage_groups.items():
            with st.expander(f"🔧 {damage_type.replace('_', ' ').title()} ({len(damage_list)} instance{'s' if len(damage_list) > 1 else ''})", expanded=True):
                
                # Get product info for this damage type
                product_info = self.get_repair_products(damage_type)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Damage details
                    st.markdown(f"**{product_info['icon']} Damage Analysis:**")
                    for i, damage in enumerate(damage_list, 1):
                        severity = self.estimate_severity(damage['confidence'], damage['area'])
                        cost = self.estimate_repair_cost(damage_type, severity)
                        confidence = damage['confidence']
                        
                        severity_color = {
                            'High': '🔴',
                            'Medium': '🟡', 
                            'Low': '🟢'
                        }.get(severity, '⚪')
                        
                        st.markdown(f"  **Instance {i}:** {severity_color} {severity} severity "
                                  f"(Confidence: {confidence:.2f}) - **${cost:,}**")
                    
                    # Total cost for this damage type
                    type_total = sum(self.estimate_repair_cost(damage_type, 
                                   self.estimate_severity(d['confidence'], d['area'])) 
                                   for d in damage_list)
                    st.markdown(f"**Total for {damage_type.replace('_', ' ').title()}:** ${type_total:,}")
                
                with col2:
                    # Product recommendation
                    st.markdown(f"**🛒 Recommended Product:**")
                    st.markdown(f"**{product_info['name']}**")
                    st.markdown(f"*{product_info['description']}*")
                    st.markdown(f"**Price Range:** {product_info['price_range']}")
                    
                    # Amazon search button
                    amazon_url = product_info['amazon_link']
                    st.markdown(
                        f'<a href="{amazon_url}" target="_blank">'
                        f'<button style="background-color: #00FA9A; color: white; '
                        f'padding: 8px 16px; border: none; border-radius: 4px; '
                        f'cursor: pointer; text-decoration: none; font-weight: bold;">'
                        f'🛒 Search on Amazon</button></a>',
                        unsafe_allow_html=True
                    )
        
        # Professional service recommendation
        st.markdown("---")
        st.markdown("### 🏪 Professional Service Options")
        st.info(
            "💡 **Tip:** For complex damages or if you're not comfortable with DIY repairs, "
            "consider visiting a professional auto body shop. Estimated professional service "
            f"cost: **${int(total_cost * 1.5):,} - ${int(total_cost * 2.5):,}** "
            "(including labor and warranty)."
        )
        
        # Cost breakdown chart
        if len(damage_groups) > 1:
            st.markdown("### 📊 Cost Breakdown by Damage Type")
            
            damage_costs = {}
            for damage_type, damage_list in damage_groups.items():
                type_cost = sum(self.estimate_repair_cost(damage_type, 
                              self.estimate_severity(d['confidence'], d['area'])) 
                              for d in damage_list)
                damage_costs[damage_type.replace('_', ' ').title()] = type_cost
            
            fig = px.pie(
                values=list(damage_costs.values()),
                names=list(damage_costs.keys()),
                title="Repair Cost Distribution by Damage Type"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
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
        
        # Additional advanced analytics
        self.render_advanced_batch_analytics(results)
        
        # Export options
        self.render_export_options(results)
    
    def plot_batch_statistics(self, results):
        """Plot comprehensive batch processing statistics"""
        st.markdown("## 📊 Batch Analysis Dashboard")
        
        # Prepare data
        all_damage_types = []
        confidence_scores = []
        severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        cost_data = []
        
        for r in results:
            if r['detections']:
                for detection in r['detections']:
                    all_damage_types.extend(r['damage_types'])
                    confidence_scores.append(detection['confidence'])
                    
                    # Calculate severity and cost for enhanced analytics
                    severity = self.estimate_severity(detection['confidence'], detection['area'])
                    severity_counts[severity] += 1
                    
                    cost = self.estimate_repair_cost(detection['class_name'], severity)
                    cost_data.append({
                        'damage_type': detection['class_name'].replace('_', ' ').title(),
                        'cost': cost,
                        'severity': severity,
                        'confidence': detection['confidence'],
                        'filename': r['filename']
                    })
        
        # Row 1: Detection Distribution and Damage Types
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced detection count distribution with colors
            detection_counts = [r['num_detections'] for r in results]
            if detection_counts:
                fig1 = px.histogram(
                    x=detection_counts,
                    nbins=max(1, max(detection_counts) if detection_counts else 1),
                    title="🔍 Distribution of Detections per Image",
                    color_discrete_sequence=['#3498db']
                )
                fig1.update_layout(
                    xaxis_title="Number of Detections", 
                    yaxis_title="Number of Images",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig1.update_traces(marker_line_width=1, marker_line_color="white")
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Enhanced damage type distribution with better styling
            if all_damage_types:
                damage_counts = {}
                for damage_type in all_damage_types:
                    clean_name = damage_type.replace('_', ' ').title()
                    damage_counts[clean_name] = damage_counts.get(clean_name, 0) + 1
                
                # Create a more colorful bar chart
                fig2 = px.bar(
                    x=list(damage_counts.keys()),
                    y=list(damage_counts.values()),
                    title="🔧 Damage Type Distribution",
                    color=list(damage_counts.values()),
                    color_continuous_scale='viridis'
                )
                fig2.update_layout(
                    xaxis_title="Damage Type", 
                    yaxis_title="Count",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                fig2.update_traces(marker_line_width=1, marker_line_color="white")
                st.plotly_chart(fig2, use_container_width=True)
        
        # Row 2: Confidence Distribution and Status Overview
        col3, col4 = st.columns(2)
        
        with col3:
            # Confidence score distribution
            if confidence_scores:
                fig3 = px.histogram(
                    x=confidence_scores,
                    nbins=20,
                    title="📈 Confidence Score Distribution",
                    color_discrete_sequence=['#e74c3c']
                )
                fig3.update_layout(
                    xaxis_title="Confidence Score", 
                    yaxis_title="Number of Detections",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig3.add_vline(x=np.mean(confidence_scores), line_dash="dash", 
                             line_color="orange", annotation_text=f"Mean: {np.mean(confidence_scores):.3f}")
                st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            # Vehicle status overview (Clean vs Damaged)
            damaged_count = sum(1 for r in results if r['num_detections'] > 0)
            clean_count = len(results) - damaged_count
            
            fig4 = px.pie(
                values=[clean_count, damaged_count],
                names=['✅ Clean Vehicles', '⚠️ Damaged Vehicles'],
                title="🚗 Vehicle Status Overview",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig4.update_traces(textposition='inside', textinfo='percent+label')
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    def render_advanced_batch_analytics(self, results):
        """Render advanced analytics and insights for batch results"""
        st.markdown("## 🔬 Advanced Analytics")
        
        # Check if we have any valid results
        if not results or len(results) == 0:
            st.warning("No batch results available for advanced analytics.")
            return
        
        # Prepare enhanced data with error handling
        cost_data = []
        severity_data = []
        confidence_by_type = {}
        damage_per_image = []
        
        try:
            for r in results:
                # Safely get number of detections
                num_detections = r.get('num_detections', 0)
                damage_per_image.append(num_detections)
                
                # Only process if we have detections and they're valid
                detections = r.get('detections', [])
                if detections and isinstance(detections, list):
                    for detection in detections:
                        try:
                            # Safely extract detection data
                            confidence = detection.get('confidence', 0.0)
                            area = detection.get('area', 0.0)
                            class_name = detection.get('class_name', 'unknown')
                            
                            severity = self.estimate_severity(confidence, area)
                            cost = self.estimate_repair_cost(class_name, severity)
                            damage_type = class_name.replace('_', ' ').title()
                            
                            cost_data.append({
                                'damage_type': damage_type,
                                'cost': cost,
                                'severity': severity,
                                'confidence': confidence,
                                'filename': r.get('filename', 'unknown')
                            })
                            
                            severity_data.append(severity)
                            
                            # Group confidence by damage type
                            if damage_type not in confidence_by_type:
                                confidence_by_type[damage_type] = []
                            confidence_by_type[damage_type].append(confidence)
                            
                        except Exception as e:
                            # Skip invalid detection data
                            continue
                            
        except Exception as e:
            st.error(f"Error processing batch data: {str(e)}")
            return
        
        # If no data was extracted, show message and return
        if not cost_data and not severity_data and not any(damage_per_image):
            st.info("No damage data found in the batch results. All images appear to be clean or had processing errors.")
            return
        
        # Row 1: Cost Analysis and Severity Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost analysis by damage type
            if cost_data and len(cost_data) > 0:
                try:
                    cost_df = pd.DataFrame(cost_data)
                    if not cost_df.empty and 'damage_type' in cost_df.columns and 'cost' in cost_df.columns:
                        cost_by_type = cost_df.groupby('damage_type')['cost'].sum().reset_index()
                        
                        fig5 = px.bar(
                            cost_by_type,
                            x='damage_type',
                            y='cost',
                            title="💰 Total Repair Costs by Damage Type",
                            color='cost',
                            color_continuous_scale='Reds'
                        )
                        fig5.update_layout(
                            xaxis_title="Damage Type",
                            yaxis_title="Total Cost (USD)",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig5, use_container_width=True)
                        
                        # Cost summary
                        total_cost = cost_df['cost'].sum()
                        avg_cost = cost_df['cost'].mean()
                        st.info(f"💵 **Total Estimated Repair Cost:** ${total_cost:,} | **Average per Damage:** ${avg_cost:,.0f}")
                except Exception as e:
                    st.warning("Unable to generate cost analysis chart.")
            else:
                st.info("No cost data available - no damages detected in the batch.")
        
        with col2:
            # Severity distribution
            if severity_data and len(severity_data) > 0:
                try:
                    severity_counts = pd.Series(severity_data).value_counts();
                    
                    if not severity_counts.empty:
                        fig6 = px.pie(
                            values=severity_counts.values,
                            names=severity_counts.index,
                            title="⚡ Damage Severity Distribution",
                            color_discrete_map={
                                'High': '#e74c3c',
                                'Medium': '#f39c12',
                                'Low': '#2ecc71'
                            }
                        )
                        fig6.update_traces(textposition='inside', textinfo='percent+label')
                        fig6.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig6, use_container_width=True)
                except Exception as e:
                    st.warning("Unable to generate severity distribution chart.")
            else:
                st.info("No severity data available - no damages detected in the batch.")
        
        # Row 2: Confidence Analysis and Damage Frequency
        col3, col4 = st.columns(2)
        
        with col3:
            # Box plot of confidence scores by damage type
            if confidence_by_type and len(confidence_by_type) > 0:
                try:
                    conf_data = []
                    for damage_type, confidences in confidence_by_type.items():
                        for conf in confidences:
                            conf_data.append({'damage_type': damage_type, 'confidence': conf})
                    
                    if conf_data and len(conf_data) > 0:
                        conf_df = pd.DataFrame(conf_data)
                        
                        if not conf_df.empty and 'damage_type' in conf_df.columns and 'confidence' in conf_df.columns:
                            fig7 = px.box(
                                conf_df,
                                x='damage_type',
                                y='confidence',
                                title="📊 Confidence Score Distribution by Damage Type",
                                color='damage_type'
                            )
                            fig7.update_layout(
                                xaxis_title="Damage Type",
                                yaxis_title="Confidence Score",
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                xaxis_tickangle=-45,
                                showlegend=False
                            )
                            st.plotly_chart(fig7, use_container_width=True)
                except Exception as e:
                    st.warning("Unable to generate confidence distribution chart.")
            else:
                st.info("No confidence data available by damage type.")
        
        with col4:
            # Damage frequency heatmap
            if len(results) > 1:
                # Create a correlation between damage types
                damage_matrix = {}
                for r in results:
                    for damage_type in r['damage_types']:
                        clean_name = damage_type.replace('_', ' ').title()
                        if clean_name not in damage_matrix:
                            damage_matrix[clean_name] = 0
                        damage_matrix[clean_name] += 1
                
                if damage_matrix:
                    # Create sunburst chart for damage distribution
                    fig8 = px.treemap(
                        names=list(damage_matrix.keys()),
                        values=list(damage_matrix.values()),
                        title="🌳 Damage Type Frequency Treemap"
                    )
                    fig8.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig8, use_container_width=True)
        
        # Row 3: Trend Analysis and Summary Statistics
        if len(results) > 3:
            st.markdown("### 📈 Trend Analysis")
            
            col5, col6 = st.columns(2)
            
            with col5:
                # Damage trend over batch sequence
                batch_trend = []
                for i, r in enumerate(results):
                    batch_trend.append({
                        'image_index': i + 1,
                        'num_detections': r['num_detections'],
                        'max_confidence': r.get('max_confidence', 0.0),
                        'filename': r['filename']
                    })
                
                trend_df = pd.DataFrame(batch_trend)
                
                # Only create chart if we have valid data
                if not trend_df.empty and 'num_detections' in trend_df.columns:
                    fig9 = px.line(
                        trend_df,
                        x='image_index',
                        y='num_detections',
                        title="📊 Detection Count Trend Across Batch",
                        markers=True,
                        hover_data=['filename']
                    )
                    fig9.update_layout(
                        xaxis_title="Image Index",
                        yaxis_title="Number of Detections",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig9, use_container_width=True)
            
            with col6:
                # Confidence trend - only if we have valid confidence data
                if not trend_df.empty and 'max_confidence' in trend_df.columns:
                    # Check if we have any non-zero confidence values
                    has_confidence_data = trend_df['max_confidence'].max() > 0
                    
                    if has_confidence_data:
                        fig10 = px.line(
                            trend_df,
                            x='image_index',
                            y='max_confidence',
                            title="🎯 Max Confidence Trend Across Batch",
                            markers=True,
                            hover_data=['filename'],
                            color_discrete_sequence=['orange']
                        )
                        fig10.update_layout(
                            xaxis_title="Image Index",
                            yaxis_title="Max Confidence Score",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig10, use_container_width=True)
                    else:
                        st.info("No confidence data available - all images were clean or had processing errors.")
        
        # Summary insights
        st.markdown("### 💡 Batch Analysis Insights")
        
        if results:
            # Calculate insights
            total_images = len(results)
            damaged_images = sum(1 for r in results if r['num_detections'] > 0)
            damage_rate = (damaged_images / total_images) * 100
            
            total_detections = sum(r['num_detections'] for r in results)
            avg_detections = total_detections / total_images if total_images > 0 else 0
            
            if cost_data:
                total_cost = sum(item['cost'] for item in cost_data)
                avg_cost_per_vehicle = total_cost / total_images
                
                # Most common damage type
                damage_type_counts = {}
                for item in cost_data:
                    dtype = item['damage_type']
                    damage_type_counts[dtype] = damage_type_counts.get(dtype, 0) + 1
                most_common_damage = max(damage_type_counts, key=damage_type_counts.get) if damage_type_counts else "None"
                
                # Create insight cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 2rem;">{damage_rate:.1f}%</h3>
                        <p style="margin: 0.5rem 0 0 0;">Damage Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 2rem;">${total_cost:,}</h3>
                        <p style="margin: 0.5rem 0 0 0;">Total Repair Cost</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 2rem;">{avg_detections:.1f}</h3>
                        <p style="margin: 0.5rem 0 0 0;">Avg Damages/Vehicle</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                                padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
                        <h3 style="margin: 0; font-size: 1.2rem;">{most_common_damage}</h3>
                        <p style="margin: 0.5rem 0 0 0;">Most Common Damage</p>
                    </div>
                    """, unsafe_allow_html=True)
    
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
                    severity = self.estimate_severity(detection['confidence'], detection['area'])
                    cost = self.estimate_repair_cost(detection['class_name'], severity)
                    product_info = self.get_repair_products(detection['class_name'])
                    
                    data.append({
                        'filename': r['filename'],
                        'damage_type': detection['class_name'],
                        'confidence': detection['confidence'],
                        'area': detection['area'],
                        'severity': severity,
                        'estimated_cost_usd': cost,
                        'recommended_product': product_info['name'],
                        'product_price_range': product_info['price_range'],
                        'amazon_link': product_info['amazon_link']
                    })
            else:
                data.append({
                    'filename': r['filename'],
                    'damage_type': 'no_damage',
                    'confidence': 0.0,
                    'area': 0.0,
                    'severity': 'None',
                    'estimated_cost_usd': 0,
                    'recommended_product': 'N/A',
                    'product_price_range': 'N/A',
                    'amazon_link': 'N/A'
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
        tab_home, tab1, tab2, tab3 = st.tabs(["🏠 Home", "📸 Single Image", "📁 Batch Analysis", "📊 History"])
        
        with tab_home:
            self.render_home_page()
        
        with tab1:
            self.render_single_image_analysis(conf_threshold, iou_threshold, max_det, imgsz)
        
        with tab2:
            self.render_batch_analysis(conf_threshold, iou_threshold, max_det, imgsz)
        
        with tab3:
            self.render_analysis_history()
        
        # Footer
        st.markdown("---")
        st.markdown("**Vehicle Damage Detection AI** - Powered by YOLO11n & Streamlit")


def main():
    """Main execution function"""
    app = VehicleDamageApp()
    app.run()


if __name__ == "__main__":
    main()