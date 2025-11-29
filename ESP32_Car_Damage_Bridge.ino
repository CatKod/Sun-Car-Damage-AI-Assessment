/*
 * ESP32-CAM Car Damage Detection Bridge
 * ====================================
 * 
 * This code creates a bridge between STM32F103C8T6 and Flask AI server
 * for real-time car damage detection using ESP32-CAM module.
 * 
 * Features:
 * - WiFi connectivity for AI server communication
 * - Camera capture and image upload to Flask server
 * - UART communication with STM32 microcontroller
 * - JSON response processing and forwarding
 * - Automatic retry and error handling
 * - Status LED indicators
 * 
 * Hardware: ESP32-CAM (AI-Thinker)
 * Author: AI Assistant
 * Date: November 2025
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Base64.h>

// ======================== WiFi Configuration ========================
const char* ssid = "YOUR_WIFI_NAME";          // Replace with your WiFi SSID
const char* password = "YOUR_WIFI_PASSWORD";   // Replace with your WiFi password
const char* flask_server = "http://192.168.1.100:5000";  // Replace with your PC IP

// ======================== Pin Definitions ========================
#define FLASH_LED_PIN 4     // Built-in flash LED
#define STATUS_LED_PIN 33   // Status LED (if available)

// ======================== Camera Configuration ========================
// AI-Thinker ESP32-CAM pin configuration
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

// ======================== Global Variables ========================
bool wifi_connected = false;
bool camera_initialized = false;
unsigned long last_heartbeat = 0;
unsigned long last_analysis_request = 0;
int analysis_counter = 0;
String latest_result = "";

// Camera and HTTP objects
camera_fb_t * fb = NULL;
HTTPClient http;
WiFiClient client;

// ======================== Function Declarations ========================
void setup();
void loop();
bool initializeWiFi();
bool initializeCamera();
bool performDamageAnalysis();
bool uploadImageForAnalysis();
void handleSTM32Commands();
void sendToSTM32(String message);
void blinkLED(int pin, int times, int delayMs);
void updateStatusLED();
String parseFlaskResponse(String response);

// ======================== Setup Function ========================
void setup() {
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(FLASH_LED_PIN, OUTPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);
  digitalWrite(STATUS_LED_PIN, LOW);
  
  Serial.println("\n=======================================");
  Serial.println("🚗 ESP32-CAM Car Damage Detection Bridge");
  Serial.println("=======================================");
  
  // Startup LED sequence
  blinkLED(STATUS_LED_PIN, 3, 200);
  
  // Initialize WiFi
  Serial.println("🌐 Initializing WiFi...");
  wifi_connected = initializeWiFi();
  
  if (wifi_connected) {
    Serial.println("✅ WiFi connected successfully");
    Serial.print("📡 IP Address: ");
    Serial.println(WiFi.localIP());
    blinkLED(STATUS_LED_PIN, 2, 500);
  } else {
    Serial.println("❌ WiFi connection failed");
    blinkLED(STATUS_LED_PIN, 5, 100);  // Fast blink = error
  }
  
  // Initialize camera
  Serial.println("📸 Initializing camera...");
  camera_initialized = initializeCamera();
  
  if (camera_initialized) {
    Serial.println("✅ Camera initialized successfully");
    blinkLED(FLASH_LED_PIN, 2, 300);
  } else {
    Serial.println("❌ Camera initialization failed");
    blinkLED(FLASH_LED_PIN, 5, 100);
  }
  
  // System ready
  if (wifi_connected && camera_initialized) {
    Serial.println("🚀 System ready for car damage detection!");
    sendToSTM32("SYSTEM_READY");
  } else {
    Serial.println("⚠️  System initialization incomplete");
    sendToSTM32("SYSTEM_ERROR");
  }
  
  Serial.println("=======================================\n");
}

// ======================== Main Loop ========================
void loop() {
  // Handle incoming commands from STM32
  handleSTM32Commands();
  
  // Update status LED
  updateStatusLED();
  
  // Heartbeat every 10 seconds
  if (millis() - last_heartbeat > 10000) {
    last_heartbeat = millis();
    
    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
      wifi_connected = false;
      Serial.println("⚠️  WiFi disconnected, attempting reconnection...");
      wifi_connected = initializeWiFi();
    }
    
    // Send heartbeat to STM32
    if (wifi_connected && camera_initialized) {
      sendToSTM32("HEARTBEAT_OK");
    } else {
      sendToSTM32("HEARTBEAT_ERROR");
    }
  }
  
  delay(100);  // Small delay to prevent watchdog issues
}

// ======================== WiFi Functions ========================
bool initializeWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    return true;
  } else {
    Serial.println(" Failed!");
    return false;
  }
}

// ======================== Camera Functions ========================
bool initializeCamera() {
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
  
  // Frame size configuration
  if (psramFound()) {
    config.frame_size = FRAMESIZE_SVGA;  // 800x600
    config.jpeg_quality = 10;            // Lower = better quality
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_CIF;   // 352x288
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }
  
  // Camera sensor adjustments for better car damage detection
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 1);       // -2 to 2  
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // 0 to 6 (0=No Effect, 1=Negative, 2=Grayscale, 3=Red Tint, 4=Green Tint, 5=Blue Tint, 6=Sepia)
    s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
    s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
    s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
    s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
    s->set_aec2(s, 0);           // 0 = disable , 1 = enable
    s->set_ae_level(s, 0);       // -2 to 2
    s->set_aec_value(s, 300);    // 0 to 1200
    s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
    s->set_bpc(s, 0);            // 0 = disable , 1 = enable
    s->set_wpc(s, 1);            // 0 = disable , 1 = enable
    s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
    s->set_lenc(s, 1);           // 0 = disable , 1 = enable
    s->set_hmirror(s, 0);        // 0 = disable , 1 = enable
    s->set_vflip(s, 0);          // 0 = disable , 1 = enable
    s->set_dcw(s, 1);            // 0 = disable , 1 = enable
    s->set_colorbar(s, 0);       // 0 = disable , 1 = enable
  }
  
  return true;
}

// ======================== Damage Analysis Functions ========================
bool performDamageAnalysis() {
  if (!camera_initialized || !wifi_connected) {
    Serial.println("❌ Cannot perform analysis: camera or WiFi not ready");
    sendToSTM32("{\"status\":\"error\",\"message\":\"System not ready\"}");
    return false;
  }
  
  Serial.println("📸 Starting damage analysis...");
  
  // Flash LED to indicate capture
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(100);
  
  // Capture image
  fb = esp_camera_fb_get();
  digitalWrite(FLASH_LED_PIN, LOW);
  
  if (!fb) {
    Serial.println("❌ Camera capture failed");
    sendToSTM32("{\"status\":\"error\",\"message\":\"Camera capture failed\"}");
    return false;
  }
  
  Serial.printf("📷 Image captured: %dx%d, %d bytes\n", 
                fb->width, fb->height, fb->len);
  
  // Upload to Flask server
  bool success = uploadImageForAnalysis();
  
  // Return frame buffer
  esp_camera_fb_return(fb);
  fb = NULL;
  
  analysis_counter++;
  last_analysis_request = millis();
  
  return success;
}

bool uploadImageForAnalysis() {
  if (!fb) {
    Serial.println("❌ No image to upload");
    return false;
  }
  
  Serial.println("🌐 Uploading image to Flask server...");
  
  http.begin(client, String(flask_server) + "/upload");
  http.addHeader("Content-Type", "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW");
  
  // Create multipart form data
  String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
  String header = "--" + boundary + "\r\n";
  header += "Content-Disposition: form-data; name=\"image\"; filename=\"esp32_image.jpg\"\r\n";
  header += "Content-Type: image/jpeg\r\n\r\n";
  
  String footer = "\r\n--" + boundary + "--\r\n";
  
  // Calculate content length
  int contentLength = header.length() + fb->len + footer.length();
  http.addHeader("Content-Length", String(contentLength));
  
  // Send request
  WiFiClient* stream = http.getStreamPtr();
  
  // Send header
  stream->print(header);
  
  // Send image data in chunks
  const uint8_t* data = fb->buf;
  size_t remaining = fb->len;
  const size_t chunkSize = 1024;
  
  while (remaining > 0) {
    size_t toSend = min(remaining, chunkSize);
    stream->write(data, toSend);
    data += toSend;
    remaining -= toSend;
    delay(1);  // Small delay to prevent WDT
  }
  
  // Send footer
  stream->print(footer);
  
  // Get response
  int httpCode = http.GET();
  String response = "";
  
  if (httpCode > 0) {
    response = http.getString();
    Serial.printf("🔄 HTTP Response: %d\n", httpCode);
    
    if (httpCode == 200) {
      Serial.println("✅ Image uploaded successfully");
      
      // Parse and forward response to STM32
      String parsed_response = parseFlaskResponse(response);
      sendToSTM32(parsed_response);
      latest_result = parsed_response;
      
      http.end();
      return true;
    } else {
      Serial.printf("❌ HTTP Error: %d\n", httpCode);
      Serial.println("Response: " + response);
    }
  } else {
    Serial.printf("❌ HTTP Request failed: %s\n", http.errorToString(httpCode).c_str());
  }
  
  http.end();
  
  // Send error response to STM32
  sendToSTM32("{\"status\":\"error\",\"message\":\"Upload failed\"}");
  return false;
}

// ======================== Communication Functions ========================
void handleSTM32Commands() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    Serial.println("📨 Received from STM32: " + command);
    
    if (command == "INIT") {
      // System status check
      if (wifi_connected && camera_initialized) {
        sendToSTM32("{\"status\":\"success\",\"message\":\"System ready\"}");
      } else {
        sendToSTM32("{\"status\":\"error\",\"message\":\"System not ready\"}");
      }
    }
    else if (command == "ANALYZE") {
      // Perform damage analysis
      performDamageAnalysis();
    }
    else if (command == "STATUS") {
      // Send system status
      StaticJsonDocument<200> statusDoc;
      statusDoc["wifi_connected"] = wifi_connected;
      statusDoc["camera_ready"] = camera_initialized;
      statusDoc["analysis_count"] = analysis_counter;
      statusDoc["uptime"] = millis() / 1000;
      statusDoc["free_heap"] = ESP.getFreeHeap();
      
      String statusJson;
      serializeJson(statusDoc, statusJson);
      sendToSTM32(statusJson);
    }
    else if (command == "LATEST") {
      // Send latest result
      if (latest_result.length() > 0) {
        sendToSTM32(latest_result);
      } else {
        sendToSTM32("{\"status\":\"no_data\",\"message\":\"No analysis performed yet\"}");
      }
    }
    else {
      Serial.println("❓ Unknown command: " + command);
      sendToSTM32("{\"status\":\"error\",\"message\":\"Unknown command\"}");
    }
  }
}

void sendToSTM32(String message) {
  Serial.println("📤 Sending to STM32: " + message);
  Serial.print(message);
  Serial.print("\r\n");
  Serial.flush();
}

String parseFlaskResponse(String response) {
  // Parse JSON response from Flask server
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, response);
  
  if (error) {
    Serial.println("❌ JSON parsing failed: " + String(error.c_str()));
    return "{\"status\":\"error\",\"message\":\"JSON parse error\"}";
  }
  
  // Extract key information
  StaticJsonDocument<256> result;
  result["status"] = doc["status"] | "error";
  result["damage_detected"] = doc["damage_detected"] | false;
  result["damage_type"] = doc["damage_type"] | "Unknown";
  result["confidence"] = doc["confidence"] | 0.0;
  result["severity"] = doc["severity"] | 0;
  result["timestamp"] = doc["timestamp"] | "";
  result["message"] = doc["message"] | "";
  
  String resultJson;
  serializeJson(result, resultJson);
  
  return resultJson;
}

// ======================== Utility Functions ========================
void blinkLED(int pin, int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(pin, HIGH);
    delay(delayMs);
    digitalWrite(pin, LOW);
    delay(delayMs);
  }
}

void updateStatusLED() {
  static unsigned long lastBlink = 0;
  static bool ledState = false;
  
  unsigned long interval;
  
  if (!wifi_connected || !camera_initialized) {
    interval = 200;  // Fast blink for error
  } else {
    interval = 2000; // Slow blink for normal operation
  }
  
  if (millis() - lastBlink > interval) {
    ledState = !ledState;
    digitalWrite(STATUS_LED_PIN, ledState);
    lastBlink = millis();
  }
}