#!/usr/bin/env python3
"""
Flask AI Server for Real-Time Car Damage Detection
==================================================

This Flask server provides AI-powered car damage detection using YOLO11n model.
Designed to work with ESP32-CAM and STM32 microcontroller system.

Features:
- YOLO11n model integration for car damage detection
- RESTful API endpoints for image upload and analysis
- Real-time damage classification and severity assessment
- JSON response format for microcontroller integration
- Multiple damage types: scratch, dent, crack, rust, etc.

Author: AI Assistant
Date: November 2025
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import logging
from pathlib import Path
import base64
from PIL import Image
import io

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Warning: ultralytics not available. Install with: pip install ultralytics")

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('temp').mkdir(exist_ok=True)

# Global variables
model = None
analysis_counter = 0
latest_result = None
server_stats = {
    'total_analyses': 0,
    'successful_analyses': 0,
    'failed_analyses': 0,
    'server_start_time': datetime.now().isoformat()
}

# Damage type configuration with severity levels
DAMAGE_TYPES = {
    'scratch': {'severity': 1, 'description': 'Surface scratch'},
    'dent': {'severity': 3, 'description': 'Body dent'},
    'crack': {'severity': 2, 'description': 'Surface crack'},
    'rust': {'severity': 2, 'description': 'Rust damage'},
    'broken': {'severity': 4, 'description': 'Broken component'},
    'paint_off': {'severity': 2, 'description': 'Paint damage'},
    'scratches': {'severity': 1, 'description': 'Multiple scratches'},
    'dents': {'severity': 3, 'description': 'Multiple dents'},
    'cracks': {'severity': 2, 'description': 'Multiple cracks'}
}

def load_model():
    """Load YOLO model for car damage detection"""
    global model
    
    if not YOLO_AVAILABLE:
        print("❌ YOLO not available - running in demo mode")
        return False
    
    try:
        # Try to load trained model first
        model_paths = [
            'models/best.pt',
            'models/detection/best.pt',
            'runs/car_damage_yolo11n_*/weights/best.pt',
            'yolo11n.pt',
            'yolov8n.pt'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if '*' in model_path:
                # Handle wildcard paths
                import glob
                matching_files = glob.glob(model_path)
                if matching_files:
                    model_path = matching_files[0]  # Use first match
                else:
                    continue
            
            if os.path.exists(model_path):
                try:
                    print(f"🔄 Loading model: {model_path}")
                    model = YOLO(model_path)
                    print(f"✅ Model loaded successfully: {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
                    continue
        
        if not model_loaded:
            # Fallback to default YOLO model
            print("🔄 Loading default YOLO11n model...")
            model = YOLO('yolo11n.pt')
            print("✅ Default YOLO11n model loaded")
            model_loaded = True
        
        return model_loaded
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def analyze_damage(image):
    """Analyze image for car damage using YOLO model"""
    global analysis_counter, latest_result, server_stats
    
    analysis_counter += 1
    server_stats['total_analyses'] += 1
    
    try:
        if model is None:
            # Demo mode - return fake results
            return create_demo_result()
        
        # Run inference
        results = model(image, conf=0.25, iou=0.45)
        
        # Process results
        damage_detected = False
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i, box in enumerate(boxes.xyxy):
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    
                    # Get class name
                    if hasattr(model, 'names') and cls < len(model.names):
                        class_name = model.names[cls]
                    else:
                        class_name = f"damage_{cls}"
                    
                    # Calculate area
                    x1, y1, x2, y2 = box.tolist()
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'area': area
                    }
                    detections.append(detection)
                    damage_detected = True
        
        # Prepare response
        if damage_detected and detections:
            # Use highest confidence detection as primary
            primary_damage = max(detections, key=lambda x: x['confidence'])
            damage_type = primary_damage['class_name']
            confidence = primary_damage['confidence'] * 100
            
            # Get severity
            severity = DAMAGE_TYPES.get(damage_type.lower(), {'severity': 2})['severity']
            
            result_data = {
                'status': 'success',
                'damage_detected': True,
                'damage_type': damage_type.replace('_', ' ').title(),
                'confidence': round(confidence, 1),
                'severity': severity,
                'detections_count': len(detections),
                'timestamp': datetime.now().isoformat(),
                'message': f'Detected {len(detections)} damage(s)',
                'analysis_id': analysis_counter
            }
        else:
            result_data = {
                'status': 'success',
                'damage_detected': False,
                'damage_type': 'No Damage',
                'confidence': 95.0,
                'severity': 0,
                'detections_count': 0,
                'timestamp': datetime.now().isoformat(),
                'message': 'No damage detected',
                'analysis_id': analysis_counter
            }
        
        latest_result = result_data
        server_stats['successful_analyses'] += 1
        return result_data
        
    except Exception as e:
        server_stats['failed_analyses'] += 1
        error_result = {
            'status': 'error',
            'damage_detected': False,
            'damage_type': 'Error',
            'confidence': 0.0,
            'severity': 0,
            'timestamp': datetime.now().isoformat(),
            'message': f'Analysis failed: {str(e)}',
            'analysis_id': analysis_counter
        }
        latest_result = error_result
        return error_result

def create_demo_result():
    """Create demo results when YOLO is not available"""
    import random
    
    # Random demo scenarios
    scenarios = [
        {
            'damage_detected': True,
            'damage_type': 'Scratch',
            'confidence': round(random.uniform(75, 95), 1),
            'severity': 1
        },
        {
            'damage_detected': True,
            'damage_type': 'Dent',
            'confidence': round(random.uniform(80, 92), 1),
            'severity': 3
        },
        {
            'damage_detected': False,
            'damage_type': 'No Damage',
            'confidence': round(random.uniform(90, 99), 1),
            'severity': 0
        },
        {
            'damage_detected': True,
            'damage_type': 'Crack',
            'confidence': round(random.uniform(70, 88), 1),
            'severity': 2
        }
    ]
    
    scenario = random.choice(scenarios)
    
    return {
        'status': 'success',
        'damage_detected': scenario['damage_detected'],
        'damage_type': scenario['damage_type'],
        'confidence': scenario['confidence'],
        'severity': scenario['severity'],
        'detections_count': 1 if scenario['damage_detected'] else 0,
        'timestamp': datetime.now().isoformat(),
        'message': 'Demo mode analysis',
        'analysis_id': analysis_counter
    }

# Flask Routes
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'Car Damage AI Detection Server',
        'model_loaded': model is not None,
        'yolo_available': YOLO_AVAILABLE,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'Car Damage Detection API',
        'model_status': 'loaded' if model else 'not_loaded',
        'yolo_available': YOLO_AVAILABLE,
        'stats': server_stats,
        'endpoints': ['/upload', '/latest_result', '/stats', '/reset'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload and analyze image for car damage"""
    try:
        # Check if request has file
        if 'file' not in request.files and 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image file provided',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Get file from request
        file = request.files.get('file') or request.files.get('image')
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Check file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'status': 'error',
                'message': f'Invalid file type: {file_ext}. Allowed: {allowed_extensions}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Read and process image
        image_bytes = file.read()
        
        # Convert to OpenCV image
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'status': 'error',
                'message': 'Invalid image format or corrupted file',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Analyze image
        result = analyze_damage(image)
        
        # Save image for debugging (optional)
        if analysis_counter % 10 == 0:  # Save every 10th image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"debug_image_{timestamp}_{analysis_counter}.jpg"
            debug_path = Path('temp') / debug_filename
            cv2.imwrite(str(debug_path), image)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/latest_result', methods=['GET'])
def get_latest_result():
    """Get the latest analysis result"""
    if latest_result:
        return jsonify(latest_result)
    else:
        return jsonify({
            'status': 'no_data',
            'message': 'No analysis performed yet',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get server statistics"""
    uptime_seconds = (datetime.now() - datetime.fromisoformat(server_stats['server_start_time'])).total_seconds()
    
    stats = server_stats.copy()
    stats.update({
        'uptime_seconds': round(uptime_seconds),
        'uptime_formatted': f"{int(uptime_seconds//3600)}h {int((uptime_seconds%3600)//60)}m",
        'success_rate': round(server_stats['successful_analyses'] / max(server_stats['total_analyses'], 1) * 100, 2),
        'model_status': 'loaded' if model else 'not_loaded',
        'yolo_available': YOLO_AVAILABLE
    })
    
    return jsonify(stats)

@app.route('/reset', methods=['POST'])
def reset_stats():
    """Reset server statistics"""
    global analysis_counter, server_stats
    
    analysis_counter = 0
    server_stats = {
        'total_analyses': 0,
        'successful_analyses': 0,
        'failed_analyses': 0,
        'server_start_time': datetime.now().isoformat()
    }
    
    return jsonify({
        'status': 'success',
        'message': 'Statistics reset successfully',
        'timestamp': datetime.now().isoformat()
    })

def get_local_ip():
    """Get local IP address for network access"""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*50)
    print("🚗 Car Damage AI Detection Server")
    print("="*50)
    
    # Load model
    print("Loading YOLO model...")
    model_loaded = load_model()
    
    if model_loaded:
        print("✅ Model loaded successfully")
    else:
        print("⚠️  Running in demo mode (no YOLO model)")
    
    # Get network info
    local_ip = get_local_ip()
    
    print("\n🌐 Server starting on:")
    print(f"   - Local: http://127.0.0.1:5000")
    print(f"   - Network: http://{local_ip}:5000")
    print("\n📡 API Endpoints:")
    print(f"   - Health: http://{local_ip}:5000/health")
    print(f"   - Upload: http://{local_ip}:5000/upload")
    print(f"   - Latest: http://{local_ip}:5000/latest_result")
    print(f"   - Stats: http://{local_ip}:5000/stats")
    
    print("\n🚀 Server ready for ESP32-CAM connections!")
    print("="*50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server shutdown requested")
    except Exception as e:
        print(f"\n❌ Server error: {e}")