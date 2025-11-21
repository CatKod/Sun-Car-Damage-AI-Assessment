/*
 * ESP32-CAM Code để gửi ảnh lên Flask Server
 * 
 * Cách hoạt động:
 * 1. ESP32-CAM chụp ảnh
 * 2. Chuyển ảnh thành JPEG buffer
 * 3. POST ảnh binary đến http://192.168.1.28:5000/upload
 * 4. Nhận response JSON từ server
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

// WiFi credentials
const char* ssid = "I2";
const char* password = "abcd1232";

// Flask Server URL
// ⚠️ QUAN TRỌNG: Sử dụng IP WiFi vì ESP32-CAM kết nối qua WiFi
// IP WiFi của máy: 192.168.1.25 (kiểm tra bằng: ipconfig)
const char* serverUrl = "http://192.168.1.25:5000/upload";

// Camera pins cho ESP32-CAM AI-Thinker
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

void setup() {
  Serial.begin(115200);
  
  // Kết nối WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  
  // Cấu hình camera
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
  
  // Cấu hình chất lượng ảnh
  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA;  // 800x600
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;   // 640x480
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  
  // Khởi tạo camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    return;
  }
  
  Serial.println("Camera initialized successfully!");
}

void loop() {
  // Chụp ảnh
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    return;
  }
  
  Serial.println("Picture taken! Size: " + String(fb->len) + " bytes");
  
  // Gửi ảnh lên server
  if(WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    
    // Tăng timeout cho kết nối
    http.setTimeout(10000); // 10 giây
    
    // Kết nối đến server
    Serial.println("Connecting to: " + String(serverUrl));
    
    if(http.begin(serverUrl)) {
      http.addHeader("Content-Type", "image/jpeg");
      
      // POST ảnh binary
      Serial.println("Sending image...");
      int httpResponseCode = http.POST(fb->buf, fb->len);
      
      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("✓ Response code: " + String(httpResponseCode));
        Serial.println("✓ Response: " + response);
      } else {
        Serial.println("✗ Error code: " + String(httpResponseCode));
        
        // In chi tiết lỗi
        if(httpResponseCode == -1) {
          Serial.println("  → Connection failed. Check:");
          Serial.println("    1. Flask server is running (python app.py)");
          Serial.println("    2. Server IP is correct: " + String(serverUrl));
          Serial.println("    3. Firewall allows port 5000");
          Serial.println("    4. ESP32 and server on same network");
        }
      }
      
      http.end();
    } else {
      Serial.println("✗ Unable to connect to server!");
      Serial.println("  Check server URL: " + String(serverUrl));
    }
  } else {
    Serial.println("WiFi not connected");
  }
  
  // Giải phóng bộ nhớ
  esp_camera_fb_return(fb);
  
  // Chờ 3 giây trước khi chụp ảnh tiếp theo
  delay(500);
}

