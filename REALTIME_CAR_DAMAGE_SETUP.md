# Real-Time Car Damage Detection System - Complete Setup Guide

## 🎯 **System Overview**

Hệ thống AI nhận diện hư hại xe theo thời gian thực với 3 components chính:

```
ESP32-CAM ←→ Flask AI Server ←→ STM32F103C8T6 + LCD1602
```

### **Data Flow:**
1. **STM32** gửi lệnh "ANALYZE" → **ESP32-CAM**
2. **ESP32-CAM** chụp ảnh → gửi **Flask AI Server**  
3. **Flask AI Server** xử lý YOLO → trả kết quả
4. **ESP32-CAM** forward JSON → **STM32**
5. **STM32** parse JSON → hiển thị **LCD1602**

---

## 🔧 **Hardware Setup**

### **Component List:**
- **STM32F103C8T6** (Blue Pill)
- **LCD1602A + I2C Module** (PCF8574, addr 0x27)
- **ESP32-CAM** (AI-Thinker)
- **Jumper wires** 
- **Breadboard**
- **5V Power supply**

### **Connections:**

#### **STM32F103C8T6 ↔ LCD1602A (I2C)**
```
STM32 Pin  →  LCD I2C Module
PB6        →  SCL
PB7        →  SDA  
5V         →  VCC
GND        →  GND
```

#### **STM32F103C8T6 ↔ ESP32-CAM (UART)**
```
STM32 Pin  →  ESP32-CAM Pin
PA9 (TX1)  →  GPIO3 (RX)
PA10 (RX1) →  GPIO1 (TX)
GND        →  GND
```

#### **ESP32-CAM Power**
```
5V Power   →  ESP32-CAM 5V
GND        →  ESP32-CAM GND
```

---

## 💻 **Software Setup**

### **Step 1: Setup Flask AI Server**

#### **1.1 Install Dependencies**
```bash
# Run the setup script
run_car_damage_system.bat

# Or manually:
pip install flask ultralytics opencv-python pillow numpy requests
```

#### **1.2 Choose Server Mode**
You have 3 options for running the AI server:

**Option A: Standalone Flask Server (Recommended for ESP32)**
```bash
python flask_damage_server.py
```

**Option B: Streamlit App with Integrated Flask**
```bash
python -m streamlit run app/streamlit_app.py
```
*Choose "Real-Time Camera" tab to activate Flask server*

**Option C: Unified Launcher**
```bash
run_app.bat
```
*Choose option 2 for Real-Time Hardware System*

#### **1.3 Server Configuration**
The server will show your PC IP address. Copy this IP for ESP32 configuration.

**Expected Output:**
```
🚗 Car Damage AI Detection Server
========================================
Loading YOLO model...
✅ Model loaded successfully
🌐 Server starting on:
   - Local: http://127.0.0.1:5000
   - Network: http://192.168.1.100:5000
🚀 Server ready for ESP32-CAM connections!
```

### **Step 2: Configure ESP32-CAM**

#### **2.1 Update WiFi Credentials**
Edit `ESP32_Car_Damage_Bridge.ino`:
```cpp
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
const char* flask_server = "http://192.168.1.100:5000";  // Your PC IP
```

#### **2.2 Upload ESP32 Code**
- Use Arduino IDE
- Select **AI Thinker ESP32-CAM** board
- Install required libraries:
  ```
  ArduinoJson by Benoit Blanchon
  ESP32 Camera library (built-in)
  ```
- Upload `ESP32_Car_Damage_Bridge.ino`

#### **2.3 Verify ESP32 Serial Output**
```
=======================================
🚗 ESP32-CAM Car Damage Detection Bridge
=======================================
🌐 Initializing WiFi...
✅ WiFi connected successfully
📡 IP Address: 192.168.1.101
📸 Initializing camera...
✅ Camera initialized successfully
🚀 System ready for car damage detection!
```

### **Step 3: Configure STM32F103C8T6**

#### **3.1 Upload STM32 Code**
- Use STM32CubeIDE
- Open project: `RL_STM32_CAR`
- Compile and upload `main.c` (contains car damage detection system)

#### **3.2 Verify STM32 Boot Sequence**
```
1 long blink (1s)     → System starting
3 quick blinks (200ms) → LCD initialized successfully
LCD: "CAR DAMAGE AI" / "SYSTEM READY"
```

---

## 🧪 **System Testing**

### **Step 1: Test Flask Server**
```bash
# Run integration test
python test_system_integration.py
```

**Expected Test Results:**
```
🧪 Running comprehensive tests...
✅ Server health check passed
✅ Image upload test passed
✅ Latest result test passed
✅ ESP32 workflow simulation passed
🎉 ALL TESTS PASSED! System integration successful!
```

### **Step 2: Test ESP32-STM32 Communication**
1. **Power up STM32** → LCD shows "CAR DAMAGE AI"
2. **Power up ESP32** → Serial shows WiFi connection
3. **Watch LCD** → Should show "SYSTEM READY" / "Uptime: XXXs"
4. **Automatic Analysis** → Every 5 seconds, LCD updates with results

---

## 🚀 **System Operation**

### **Startup Sequence:**

#### **1. STM32 Boot (LED Patterns):**
```
1 long blink (1s)     → System starting
3 quick blinks (200ms) → LCD initialized successfully
```

#### **2. LCD Display Sequence:**
```
"CAR DAMAGE AI"
"SYSTEM READY"
↓ (2s later)
"INITIALIZING..."
"Please wait"
↓ (ESP32 connects)
"SYSTEM READY"
"Uptime: XXXs"
```

#### **3. Analysis Cycle (Every 5 seconds):**
```
LCD: "ANALYZING..."     →  ESP32: Capture photo
     "Please wait..."   →  Flask: AI processing
     ↓
     "SCRATCH"          ←  Results received
     "85% SEV:1"        ←  Confidence + Severity
     ↓ (or)
     "NO DAMAGE"        ←  No damage detected
     "CONF: 95%"        ←  High confidence
```

### **LED Status Indicators:**
```
🟢 Slow blink (2s)    = System ready, waiting
🟡 Fast blink (250ms) = Analyzing image  
🔴 Ultra fast (100ms) = System error
💙 Steady 1Hz         = Normal operation
```

---

## 🔍 **Troubleshooting**

### **Issue 1: Flask Server Won't Start**
**Symptoms:** 
- `ModuleNotFoundError: No module named 'ultralytics'`
- Port 5000 already in use

**Solutions:**
```bash
# Install missing packages
pip install ultralytics flask opencv-python pillow

# Kill processes using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# Try different port
# Edit flask_damage_server.py: app.run(port=5001)
```

### **Issue 2: ESP32 Cannot Connect to WiFi**
**Symptoms:**
- Serial: "WiFi connection failed"
- LCD: "SYSTEM ERROR"

**Solutions:**
1. Check WiFi credentials in ESP32 code
2. Ensure 2.4GHz WiFi (ESP32 doesn't support 5GHz)
3. Check WiFi signal strength
4. Verify ESP32-CAM power supply (needs stable 5V)

### **Issue 3: LCD Shows "SYSTEM ERROR"**
**Symptoms:**
- LCD stuck on error message
- No response to ESP32 commands

**Solutions:**
1. Check UART connections (PA9↔GPIO3, PA10↔GPIO1, GND↔GND)
2. Verify ESP32 is running and connected to WiFi
3. Test Flask server with browser: `http://YOUR_PC_IP:5000/health`
4. Check STM32 power and reset

### **Issue 4: Camera Initialization Failed**
**Symptoms:**
- ESP32 Serial: "Camera init failed"
- No image capture possible

**Solutions:**
1. Check ESP32-CAM power (needs stable 5V, 500mA+)
2. Verify camera cable connection
3. Press ESP32-CAM reset button
4. Try different ESP32-CAM module

### **Issue 5: AI Analysis Always Returns "No Damage"**
**Symptoms:**
- All images analyzed as "No Damage" 
- High confidence but incorrect results

**Solutions:**
1. Check if proper YOLO model is loaded
2. Verify image quality and lighting
3. Test with known damaged car images
4. Update YOLO model training data

---

## 📊 **Expected Results**

### **Damage Detection Types:**
- **Crack**: Severity 2 (Medium)
- **Dent**: Severity 3 (High) 
- **Scratch**: Severity 1 (Low)
- **Broken**: Severity 4 (Critical)
- **Rust**: Severity 2 (Medium)
- **Paint Off**: Severity 2 (Medium)

### **LCD Display Format:**
```
Line 1: Damage Type (max 16 chars)
Line 2: Confidence% SEV:X
        or
        CONF: XX% (for no damage)
```

### **Analysis Performance:**
- **Response Time**: 2-5 seconds per analysis
- **Update Frequency**: Every 5 seconds (configurable)
- **Network Latency**: <1 second (local network)
- **Accuracy**: Depends on YOLO model training

---

## 🌟 **Advanced Configuration**

### **Adjust Analysis Interval:**
In `main.c`:
```c
uint32_t analysis_interval = 5000; // Change to desired milliseconds
```

### **Change Camera Resolution:**
In `ESP32_Car_Damage_Bridge.ino`:
```cpp
// For better quality (larger file):
config.frame_size = FRAMESIZE_SVGA;  // 800x600
config.jpeg_quality = 8;             // Lower = better quality

// For faster upload (smaller file):
config.frame_size = FRAMESIZE_CIF;   // 352x288
config.jpeg_quality = 15;            // Higher = smaller file
```

### **Modify Damage Severity Levels:**
In `flask_damage_server.py`:
```python
DAMAGE_TYPES = {
    'scratch': {'severity': 1},  # Light damage
    'dent': {'severity': 3},     # Heavy damage
    'crack': {'severity': 2},    # Medium damage
    'broken': {'severity': 4},   # Critical damage
}
```

### **Custom LCD I2C Address:**
In `main.c`:
```c
#define LCD_I2C_ADDR 0x27  // Change if different address
```

---

## 🎯 **Performance Optimization**

### **Network Optimization:**
1. Use 2.4GHz WiFi for ESP32 (better range)
2. Place ESP32 close to WiFi router
3. Use static IP for PC if possible
4. Configure WiFi QoS for better performance

### **Image Processing Optimization:**
1. Adjust camera settings for lighting conditions
2. Use appropriate JPEG quality (balance size vs quality)
3. Consider image preprocessing (brightness, contrast)
4. Optimize YOLO model for edge deployment

### **System Reliability:**
1. Add watchdog timers for ESP32 and STM32
2. Implement automatic reconnection for WiFi
3. Add backup communication methods
4. Log system events for debugging

---

## 🛠️ **Development Extensions**

### **Additional Features to Implement:**
1. **Multiple Camera Support** - Connect multiple ESP32-CAMs
2. **Database Logging** - Store analysis results in database  
3. **Web Dashboard** - Real-time monitoring via web interface
4. **Mobile App** - Smartphone control and monitoring
5. **Cloud Integration** - Upload results to cloud storage
6. **Advanced Analytics** - Trend analysis and reporting

### **Hardware Upgrades:**
1. **Higher Resolution Camera** - Use ESP32-CAM with better sensor
2. **External Flash** - Add LED flash for low-light conditions
3. **LCD Upgrade** - Use larger OLED or TFT display
4. **Audio Alerts** - Add buzzer for damage notifications
5. **GPS Module** - Add location tracking for damage reports

---

## 📋 **System Validation Checklist**

### **Before Deployment:**
- [ ] Flask server starts without errors
- [ ] ESP32-CAM connects to WiFi successfully  
- [ ] STM32 LCD displays system information
- [ ] UART communication works between STM32 and ESP32
- [ ] Camera captures and uploads images
- [ ] AI analysis returns consistent results
- [ ] LCD displays damage analysis correctly
- [ ] System handles errors gracefully
- [ ] All LED indicators work as expected
- [ ] Integration test passes completely

### **Performance Verification:**
- [ ] Analysis completes within 5 seconds
- [ ] System operates continuously for 1+ hours
- [ ] WiFi connection remains stable
- [ ] Memory usage stays within limits
- [ ] No system crashes or resets
- [ ] Results accuracy meets requirements

---

🚗 **Happy Car Damage Detection!** 🎯

**System Status:** ✅ Ready for Production Deployment

For technical support or issues, check the troubleshooting section or create a detailed issue report with:
- System logs from all components
- Network configuration details  
- Hardware connection verification
- Test results from integration script