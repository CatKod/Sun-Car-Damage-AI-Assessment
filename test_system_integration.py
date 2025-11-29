#!/usr/bin/env python3
"""
Car Damage Detection System Integration Test
===========================================

This script tests the integration between all components:
- Flask AI Server
- ESP32-CAM Bridge simulation
- STM32 communication simulation
- Real-time damage detection workflow

Author: AI Assistant
Date: November 2025
"""

import requests
import json
import time
import os
import sys
from pathlib import Path
from PIL import Image
import io
import base64

# Test configuration
FLASK_SERVER_URL = "http://localhost:5000"
TEST_IMAGES_DIR = Path("test_images")
RESULTS_DIR = Path("test_results")

# Create directories
TEST_IMAGES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def test_server_health():
    """Test if Flask server is running and healthy"""
    print("🔍 Testing server health...")
    try:
        response = requests.get(f"{FLASK_SERVER_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy: {data.get('service', 'Unknown')}")
            print(f"   Model Status: {data.get('model_status', 'Unknown')}")
            print(f"   YOLO Available: {data.get('yolo_available', False)}")
            return True
        else:
            print(f"❌ Server health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a simple test image (car-like shape)
    img = Image.new('RGB', (640, 480), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple car shape
    # Car body
    draw.rectangle([100, 200, 540, 350], fill='red', outline='black', width=3)
    # Wheels
    draw.ellipse([120, 320, 180, 380], fill='black', outline='gray', width=2)
    draw.ellipse([460, 320, 520, 380], fill='black', outline='gray', width=2)
    # Windows
    draw.rectangle([130, 220, 510, 280], fill='lightblue', outline='black', width=2)
    
    # Add some "damage" (scratch marks)
    draw.line([(200, 250), (300, 270)], fill='brown', width=5)
    draw.line([(350, 240), (400, 260)], fill='brown', width=3)
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), "TEST CAR - DAMAGE SIMULATION", fill='black', font=font)
    
    test_image_path = TEST_IMAGES_DIR / "test_car_damage.jpg"
    img.save(test_image_path, "JPEG", quality=85)
    print(f"📸 Created test image: {test_image_path}")
    
    return test_image_path

def test_image_upload_multipart(image_path):
    """Test image upload using multipart form data (ESP32-CAM style)"""
    print(f"📤 Testing multipart image upload: {image_path.name}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(f"{FLASK_SERVER_URL}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Upload successful!")
            print(f"   Status: {data.get('status')}")
            print(f"   Damage Detected: {data.get('damage_detected', False)}")
            print(f"   Damage Type: {data.get('damage_type', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 0)}%")
            print(f"   Severity: {data.get('severity', 0)}")
            return data
        else:
            print(f"❌ Upload failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload error: {e}")
        return None

def test_image_upload_base64(image_path):
    """Test image upload using base64 encoding (alternative method)"""
    print(f"📤 Testing base64 image upload: {image_path.name}")
    
    try:
        # Encode image to base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {'image': image_data}
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(f"{FLASK_SERVER_URL}/upload", 
                               json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Base64 upload successful!")
            print(f"   Damage Type: {data.get('damage_type', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 0)}%")
            return data
        else:
            print(f"❌ Base64 upload failed: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Base64 upload error: {e}")
        return None

def test_latest_result():
    """Test getting latest analysis result (STM32 endpoint)"""
    print("📊 Testing latest result retrieval...")
    
    try:
        response = requests.get(f"{FLASK_SERVER_URL}/latest_result", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Latest result retrieved!")
            print(f"   Status: {data.get('status')}")
            
            if data.get('damage_detected'):
                print(f"   Damage Type: {data.get('damage_type')}")
                print(f"   Confidence: {data.get('confidence')}%")
                print(f"   Severity: {data.get('severity')}")
            else:
                print("   No damage detected in latest analysis")
            
            return data
        else:
            print(f"❌ Latest result failed: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Latest result error: {e}")
        return None

def test_server_stats():
    """Test server statistics endpoint"""
    print("📈 Testing server statistics...")
    
    try:
        response = requests.get(f"{FLASK_SERVER_URL}/stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Statistics retrieved!")
            print(f"   Total Analyses: {data.get('total_analyses', 0)}")
            print(f"   Successful: {data.get('successful_analyses', 0)}")
            print(f"   Failed: {data.get('failed_analyses', 0)}")
            print(f"   Success Rate: {data.get('success_rate', 0)}%")
            print(f"   Uptime: {data.get('uptime_formatted', 'Unknown')}")
            return data
        else:
            print(f"❌ Statistics failed: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Statistics error: {e}")
        return None

def simulate_esp32_workflow():
    """Simulate complete ESP32-CAM workflow"""
    print("\n" + "="*60)
    print("🤖 ESP32-CAM Workflow Simulation")
    print("="*60)
    
    # Step 1: ESP32 boots up and checks server
    print("\n1️⃣ ESP32 Boot Sequence...")
    if not test_server_health():
        print("❌ ESP32 cannot connect to Flask server!")
        return False
    
    # Step 2: STM32 sends INIT command
    print("\n2️⃣ STM32 INIT Command...")
    print("STM32 → ESP32: 'INIT'")
    print("ESP32 → STM32: 'SYSTEM_READY'")
    
    # Step 3: STM32 requests analysis
    print("\n3️⃣ STM32 Analysis Request...")
    print("STM32 → ESP32: 'ANALYZE'")
    
    # Step 4: ESP32 captures and uploads image
    print("\n4️⃣ ESP32 Image Capture & Upload...")
    test_image = create_test_image()
    result = test_image_upload_multipart(test_image)
    
    if result:
        # Step 5: ESP32 forwards result to STM32
        print("\n5️⃣ ESP32 → STM32 Result Forward...")
        print(f"ESP32 → STM32: {json.dumps(result, indent=2)}")
        
        # Step 6: STM32 requests latest result for LCD
        print("\n6️⃣ STM32 Latest Result Request...")
        print("STM32 → ESP32: 'LATEST'")
        latest = test_latest_result()
        
        if latest:
            # Simulate LCD display format
            print("\n📺 STM32 LCD Display:")
            print("┌" + "─"*16 + "┐")
            
            if latest.get('damage_detected'):
                damage_type = latest.get('damage_type', 'DAMAGE')[:16].ljust(16)
                confidence = latest.get('confidence', 0)
                severity = latest.get('severity', 0)
                line2 = f"{confidence:.0f}% SEV:{severity}"[:16].ljust(16)
            else:
                damage_type = "NO DAMAGE"[:16].ljust(16)
                line2 = f"CONF: {latest.get('confidence', 0):.0f}%"[:16].ljust(16)
            
            print(f"│{damage_type}│")
            print(f"│{line2}│")
            print("└" + "─"*16 + "┘")
        
        return True
    else:
        print("❌ Image analysis failed!")
        return False

def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("📋 Generating Test Report")
    print("="*60)
    
    report = {
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'server_health': None,
        'image_upload_test': None,
        'latest_result_test': None,
        'server_stats': None,
        'esp32_workflow': None
    }
    
    # Run all tests
    print("\n🧪 Running comprehensive tests...")
    
    # Health check
    report['server_health'] = test_server_health()
    
    # Image upload test
    if report['server_health']:
        test_image = create_test_image()
        upload_result = test_image_upload_multipart(test_image)
        report['image_upload_test'] = upload_result is not None
        
        # Latest result test
        report['latest_result_test'] = test_latest_result() is not None
        
        # Server stats
        report['server_stats'] = test_server_stats()
        
        # Full workflow
        print("\n🔄 Running complete workflow test...")
        report['esp32_workflow'] = simulate_esp32_workflow()
    
    # Save report
    report_file = RESULTS_DIR / f"integration_test_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   Server Health: {'✅' if report['server_health'] else '❌'}")
    print(f"   Image Upload: {'✅' if report['image_upload_test'] else '❌'}")
    print(f"   Latest Result: {'✅' if report['latest_result_test'] else '❌'}")
    print(f"   ESP32 Workflow: {'✅' if report['esp32_workflow'] else '❌'}")
    
    print(f"\n📄 Report saved: {report_file}")
    
    # Overall status
    all_passed = all([
        report['server_health'],
        report['image_upload_test'], 
        report['latest_result_test'],
        report['esp32_workflow']
    ])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! System integration successful!")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
    
    return all_passed

def main():
    """Main test execution"""
    print("🚗 Car Damage Detection System Integration Test")
    print("=" * 50)
    print("This script tests the complete system integration:")
    print("• Flask AI Server connectivity")
    print("• ESP32-CAM image upload simulation")  
    print("• STM32 communication workflow")
    print("• Real-time damage detection pipeline")
    print("=" * 50)
    
    # Check if server is running
    print("\n⚡ Prerequisites Check:")
    print("1. Flask server should be running on localhost:5000")
    print("2. Run: python flask_damage_server.py")
    print("3. Or: python -m streamlit run app/streamlit_app.py")
    
    input("\nPress Enter when Flask server is ready...")
    
    # Run tests
    success = generate_test_report()
    
    if success:
        print("\n🎯 Integration test completed successfully!")
        print("Your system is ready for ESP32-CAM deployment.")
    else:
        print("\n🔧 Integration test found issues.")
        print("Please fix the errors and run the test again.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())