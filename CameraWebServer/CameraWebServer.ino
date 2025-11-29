/*
 * ESP32-CAM Bridge for Car Damage Detection System
 * 
 * Chức năng:
 * 1. Chụp ảnh và gửi lên Flask Server (AI Analysis)
 * 2. Nhận kết quả phân tích từ Flask Server
 * 3. Forward kết quả đến STM32 qua UART
 * 4. Nhận lệnh từ STM32 để lấy kết quả mới nhất
 * 
 * Kết nối UART với STM32:
 * - TX (GPIO1) → STM32 RX (PA10)
 * - RX (GPIO3) → STM32 TX (PA9)
 * - GND → GND
 * 
 * Author: Car Damage AI System
 * Date: 2025
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "I2";
const char* password = "abcd1232";

// Flask Server URLs
const char* uploadUrl = "http://192.168.1.24:5000/upload";
const char* resultUrl = "http://192.168.1.24:5000/latest_result";

// Communication with STM32
#define STM32_SERIAL Serial  // Hardware Serial for STM32 communication
#define DEBUG_SERIAL Serial  // Same serial for debug (can be changed)
#define UART_BAUD_RATE 115200

// Camera pins for ESP32-CAM AI-Thinker
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// System state
struct SystemState {
  bool camera_ready;
  bool wifi_connected;
  bool server_available;
  String latest_result;
  unsigned long last_photo_time;
  unsigned long last_result_check;
  unsigned long last_heartbeat;
} sysState;

// Timing configuration
const unsigned long PHOTO_INTERVAL = 333;      // Take photo every 2 seconds
const unsigned long RESULT_CHECK_INTERVAL = 1000; // Check results every 1 second
const unsigned long HEARTBEAT_INTERVAL = 30000;   // Heartbeat every 30 seconds

void setup() {
  // Initialize Serial communication
  STM32_SERIAL.begin(UART_BAUD_RATE);
  delay(1000);
  
  DEBUG_SERIAL.println("ESP32-CAM Car Damage Detection Bridge Starting...");
  
  // Initialize system state
  sysState.camera_ready = false;
  sysState.wifi_connected = false;
  sysState.server_available = false;
  sysState.latest_result = "";
  sysState.last_photo_time = 0;
  sysState.last_result_check = 0;
  sysState.last_heartbeat = 0;
  
  // Initialize WiFi
  initWiFi();
  
  // Initialize Camera
  initCamera();
  
  // Send initial status to STM32
  sendToSTM32("SYSTEM_READY");
  
  DEBUG_SERIAL.println("Setup completed. Bridge ready.");
}

void loop() {
  unsigned long now = millis();
  
  // Handle WiFi reconnection
  if (WiFi.status() != WL_CONNECTED) {
    sysState.wifi_connected = false;
    if (now - sysState.last_heartbeat > 5000) { // Try reconnect every 5 seconds
      DEBUG_SERIAL.println("WiFi disconnected, attempting reconnect...");
      WiFi.reconnect();
      sysState.last_heartbeat = now;
    }
  } else if (!sysState.wifi_connected) {
    sysState.wifi_connected = true;
    DEBUG_SERIAL.println("WiFi reconnected: " + WiFi.localIP().toString());
  }
  
  // Main operations (only when WiFi is connected)
  if (sysState.wifi_connected) {
    
    // 1. Take and upload photo periodically
    if (now - sysState.last_photo_time > PHOTO_INTERVAL) {
      takeAndUploadPhoto();
      sysState.last_photo_time = now;
    }
    
    // 2. Check for latest results periodically  
    if (now - sysState.last_result_check > RESULT_CHECK_INTERVAL) {
      checkLatestResult();
      sysState.last_result_check = now;
    }
    
    // 3. Send heartbeat to STM32
    if (now - sysState.last_heartbeat > HEARTBEAT_INTERVAL) {
      sendHeartbeat();
      sysState.last_heartbeat = now;
    }
  }
  
  // 4. Handle commands from STM32
  handleSTM32Commands();
  
  // Small delay to prevent watchdog issues
  delay(50);
}

void initWiFi() {
  DEBUG_SERIAL.print("Connecting to WiFi: ");
  DEBUG_SERIAL.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    DEBUG_SERIAL.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    sysState.wifi_connected = true;
    DEBUG_SERIAL.println();
    DEBUG_SERIAL.println("WiFi connected!");
    DEBUG_SERIAL.print("IP address: ");
    DEBUG_SERIAL.println(WiFi.localIP());
    
    sendToSTM32("WIFI_CONNECTED");
  } else {
    DEBUG_SERIAL.println();
    DEBUG_SERIAL.println("WiFi connection failed!");
    sendToSTM32("WIFI_FAILED");
  }
}

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA;  // 800x600
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;   // 640x480
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }
  
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    DEBUG_SERIAL.printf("Camera init failed: 0x%x\\n", err);
    sendToSTM32("CAMERA_FAILED");
    sysState.camera_ready = false;
  } else {
    DEBUG_SERIAL.println("Camera initialized successfully!");
    sendToSTM32("CAMERA_READY");
    sysState.camera_ready = true;
  }
}

void takeAndUploadPhoto() {
  if (!sysState.camera_ready) return;
  
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    DEBUG_SERIAL.println("Camera capture failed");
    return;
  }
  
  DEBUG_SERIAL.printf("[%lu] Photo taken, size: %d bytes\\n", millis(), fb->len);
  
  // Upload to Flask server
  HTTPClient http;
  http.setTimeout(5000);
  
  if(http.begin(uploadUrl)) {
    http.addHeader("Content-Type", "image/jpeg");
    
    int httpResponseCode = http.POST(fb->buf, fb->len);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      DEBUG_SERIAL.printf("Upload response [%d]: %s\\n", httpResponseCode, response.c_str());
      
      // Update server availability status
      sysState.server_available = true;
      
    } else {
      DEBUG_SERIAL.printf("Upload failed: %d\\n", httpResponseCode);
      sysState.server_available = false;
    }
    
    http.end();
  } else {
    DEBUG_SERIAL.println("Failed to connect to upload server");
    sysState.server_available = false;
  }
  
  esp_camera_fb_return(fb);
}

void checkLatestResult() {
  if (!sysState.server_available) return;
  
  HTTPClient http;
  http.setTimeout(3000);
  
  if(http.begin(resultUrl)) {
    int httpResponseCode = http.GET();
    
    if (httpResponseCode == 200) {
      String response = http.getString();
      
      // Only forward if result has changed
      if (response != sysState.latest_result) {
        sysState.latest_result = response;
        
        DEBUG_SERIAL.println("New result from server:");
        DEBUG_SERIAL.println(response);
        
        // Forward to STM32
        sendResultToSTM32(response);
      }
      
    } else {
      DEBUG_SERIAL.printf("Result check failed: %d\\n", httpResponseCode);
    }
    
    http.end();
  }
}

void sendResultToSTM32(String jsonResult) {
  // Send the complete JSON response to STM32
  // STM32 will parse it according to its parsing logic
  
  STM32_SERIAL.println(jsonResult);
  DEBUG_SERIAL.println("Sent to STM32: " + jsonResult);
}

void sendToSTM32(String message) {
  STM32_SERIAL.println(message);
  DEBUG_SERIAL.println("Status to STM32: " + message);
}

void sendHeartbeat() {
  String status = "HEARTBEAT:";
  status += WiFi.status() == WL_CONNECTED ? "WIFI_OK," : "WIFI_FAIL,";
  status += sysState.camera_ready ? "CAM_OK," : "CAM_FAIL,";
  status += sysState.server_available ? "SERVER_OK" : "SERVER_FAIL";
  
  sendToSTM32(status);
}

void handleSTM32Commands() {
  if (STM32_SERIAL.available()) {
    String command = STM32_SERIAL.readStringUntil('\\n');
    command.trim();
    
    DEBUG_SERIAL.println("Command from STM32: " + command);
    
    if (command == "GET_RESULT") {
      // STM32 is requesting the latest result
      if (sysState.latest_result.length() > 0) {
        sendResultToSTM32(sysState.latest_result);
      } else {
        // No result available, force a check
        checkLatestResult();
        if (sysState.latest_result.length() > 0) {
          sendResultToSTM32(sysState.latest_result);
        } else {
          // Send no data response
          String noDataResponse = R"({"status":"no_data","display_line1":"  NO DATA      ","display_line2":" WAITING...   "})";
          sendResultToSTM32(noDataResponse);
        }
      }
    }
    else if (command == "GET_STATUS") {
      sendHeartbeat();
    }
    else if (command == "TAKE_PHOTO") {
      // Force take a photo
      takeAndUploadPhoto();
    }
    else {
      DEBUG_SERIAL.println("Unknown command: " + command);
    }
  }
}