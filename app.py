"""
Flask Server cho ESP32-CAM Face Detection
Server nhận ảnh từ ESP32-CAM, nhận diện khuôn mặt và hiển thị stream
"""

from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
import base64
from PIL import Image
import io
import os
from datetime import datetime

app = Flask(__name__)

# Đường dẫn lưu ảnh đã nhận diện
STATIC_DIR = 'static'
DETECTED_IMAGE_PATH = os.path.join(STATIC_DIR, 'face_detected.jpg')

# Tạo thư mục static nếu chưa có
os.makedirs(STATIC_DIR, exist_ok=True)

# Load Haar Cascade cho nhận diện khuôn mặt
# Xử lý nhiều trường hợp để tương thích với các phiên bản OpenCV khác nhau
face_cascade = None
cascade_file = 'haarcascade_frontalface_default.xml'

try:
    # Cách 1: Sử dụng cv2.data (OpenCV >= 4.0)
    cascade_path = cv2.data.haarcascades + cascade_file
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print(f"✅ Loaded Haar Cascade from: {cascade_path}")
except AttributeError:
    # Cách 2: Tìm trong thư mục static
    cascade_path = os.path.join(STATIC_DIR, cascade_file)
    
    if not os.path.exists(cascade_path):
        # Tải xuống từ GitHub nếu không tìm thấy
        print("⬇️  Downloading Haar Cascade file...")
        import urllib.request
        url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_file}'
        try:
            urllib.request.urlretrieve(url, cascade_path)
            print(f"✅ Downloaded to: {cascade_path}")
        except Exception as e:
            print(f"❌ Failed to download: {e}")
            print("📥 Please download manually from:")
            print(f"   {url}")
            print(f"   and save to: {cascade_path}")
            raise
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print(f"✅ Loaded Haar Cascade from: {cascade_path}")

# Kiểm tra xem cascade có load thành công không
if face_cascade is None or face_cascade.empty():
    raise Exception("❌ Failed to load Haar Cascade classifier!")

# Biến toàn cục lưu frame mới nhất
latest_frame = None
latest_detected_frame = None


def detect_faces(image):
    """
    Nhận diện khuôn mặt trong ảnh và vẽ khung hình chữ nhật
    
    Args:
        image: numpy array của ảnh (BGR format)
    
    Returns:
        image: ảnh đã vẽ khung hình
        faces_count: số khuôn mặt phát hiện được
    """
    # Chuyển sang grayscale để nhận diện
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Nhận diện khuôn mặt
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Vẽ khung hình chữ nhật xung quanh mỗi khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Thêm thông tin số khuôn mặt và timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, f'Faces: {len(faces)} | {timestamp}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return image, len(faces)


@app.route('/')
def index():
    """Trang chủ hiển thị video stream"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Endpoint nhận ảnh từ ESP32-CAM
    ESP32-CAM có thể gửi ảnh theo 2 cách:
    1. Base64: {'image': 'base64_encoded_string'}
    2. Binary: gửi trực tiếp file trong form-data hoặc raw body
    """
    global latest_frame, latest_detected_frame
    
    try:
        image = None
        
        # Cách 1: Nhận ảnh dưới dạng base64 từ JSON
        if request.is_json:
            data = request.get_json()
            if 'image' in data:
                # Decode base64
                image_data = base64.b64decode(data['image'])
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Cách 2: Nhận ảnh dưới dạng file upload
        elif 'file' in request.files:
            file = request.files['file']
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Cách 3: Nhận ảnh dưới dạng raw binary data
        else:
            image_bytes = request.data
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'status': 'error', 'message': 'Could not decode image'}), 400
        
        # Lưu frame gốc
        latest_frame = image.copy()
        
        # Nhận diện khuôn mặt
        detected_image, faces_count = detect_faces(image)
        latest_detected_frame = detected_image.copy()
        
        # Lưu ảnh đã nhận diện vào static folder
        cv2.imwrite(DETECTED_IMAGE_PATH, detected_image)
        
        return jsonify({
            'status': 'success',
            'faces_detected': faces_count,
            'message': f'Detected {faces_count} face(s)'
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/latest')
def get_latest_image():
    """API trả về ảnh mới nhất đã nhận diện"""
    if os.path.exists(DETECTED_IMAGE_PATH):
        with open(DETECTED_IMAGE_PATH, 'rb') as f:
            image_data = f.read()
        return Response(image_data, mimetype='image/jpeg')
    else:
        return jsonify({'status': 'error', 'message': 'No image available'}), 404


@app.route('/stream')
def video_stream():
    """
    Endpoint stream video theo định dạng MJPEG
    Trả về ảnh mới nhất liên tục
    """
    def generate():
        while True:
            if latest_detected_frame is not None:
                # Encode frame thành JPEG
                ret, buffer = cv2.imencode('.jpg', latest_detected_frame)
                frame = buffer.tobytes()
                
                # Trả về frame theo định dạng MJPEG
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Nếu chưa có frame nào, tạm dừng
                import time
                time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """Kiểm tra trạng thái server"""
    return jsonify({
        'status': 'running',
        'has_frame': latest_frame is not None,
        'detected_image_exists': os.path.exists(DETECTED_IMAGE_PATH)
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Flask Face Detection Server Starting...")
    print("=" * 60)
    print(f"📡 Server URL: http://192.168.1.25:5000/")
    print(f"🌐 Web Interface: http://192.168.1.25:5000/")
    print(f"📤 Upload Endpoint: http://192.168.1.25:5000/upload")
    print(f"📺 Video Stream: http://192.168.1.25:5000/stream")
    print("=" * 60)
    print("\n⚙️  ESP32-CAM Configuration:")
    print("   - POST images to: http://192.168.1.25:5000/upload")
    print("   - Format: JPEG binary or base64 JSON")
    print(f"   - ESP32 IP: Kiểm tra Serial Monitor")
    print("=" * 60)
    
    # Chạy server trên tất cả network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
