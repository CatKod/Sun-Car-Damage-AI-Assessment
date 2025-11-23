# Hướng Dẫn Tích Hợp Hệ Thống Hiển Thị Kết Quả Lên LCD1602

## Tổng Quan Hệ Thống

```
ESP32-CAM ──[WiFi]──> Flask Server (AI) ──[HTTP]──> STM32F103C8T6 ──[I2C]──> LCD1602
    │                      │                            │
    └──────[UART]──────────┘                            │
                                                        └─> Hiển thị kết quả
```

## 1. Kết Nối Phần Cứng

### 1.1 ESP32-CAM AI-Thinker
- **Nguồn**: 5V DC (không dùng USB)
- **WiFi**: Kết nối mạng "I2" 
- **UART với STM32**:
  - TX (GPIO1) → STM32 RX (PA10)
  - RX (GPIO3) → STM32 TX (PA9)  
  - GND → GND

### 1.2 STM32F103C8T6 (Blue Pill)
- **Nguồn**: 3.3V hoặc 5V từ ST-Link
- **I2C với LCD**:
  - SCL (PB6) → LCD I2C SCL
  - SDA (PB7) → LCD I2C SDA
  - 3.3V → LCD VCC
  - GND → LCD GND
- **UART với ESP32**:
  - TX (PA9) → ESP32 RX (GPIO3)
  - RX (PA10) → ESP32 TX (GPIO1)
  - GND → ESP32 GND

### 1.3 LCD1602 với Module I2C (PCF8574)
- **VCC**: 5V hoặc 3.3V (tùy module)
- **GND**: GND
- **SDA**: Kết nối STM32 PB7
- **SCL**: Kết nối STM32 PB6
- **Địa chỉ I2C**: 0x27 hoặc 0x3F (check bằng I2C scanner)

## 2. Chuẩn Bị Phần Mềm

### 2.1 Flask Server (Đã có sẵn)
```bash
cd d:\GitHub\Sun-Car-Damage-AI-Assessment\app
python streamlit_app.py
```
- Server sẽ chạy trên: `http://192.168.1.24:5000`
- Endpoint mới: `/latest_result` để STM32 lấy kết quả

### 2.2 ESP32-CAM Code
- Sử dụng file: `ESP32_STM32_Bridge.ino`
- Cài đặt thư viện: `ArduinoJson` trong Arduino IDE
- Upload code qua USB-Serial adapter

### 2.3 STM32 Code
- Project: `RL_STM32_CAR` đã được cập nhật
- Thư viện mới: `lcd1602_i2c.c/h` và `car_damage_comm.c/h`
- Compile và upload qua ST-Link

## 3. Cấu Hình Hệ Thống

### 3.1 Kiểm Tra Địa Chỉ I2C LCD
```c
// Trong lcd1602_i2c.h, có thể cần thay đổi:
#define LCD_I2C_ADDRESS 0x27    // Hoặc 0x3F
```

### 3.2 Cấu Hình WiFi ESP32
```cpp
// Trong ESP32_STM32_Bridge.ino:
const char* ssid = "I2";              // Tên WiFi
const char* password = "abcd1232";     // Mật khẩu WiFi
const char* uploadUrl = "http://192.168.1.24:5000/upload";        // IP máy chạy Flask
const char* resultUrl = "http://192.168.1.24:5000/latest_result"; // Endpoint mới
```

### 3.3 Cấu Hình UART
- **Baud rate**: 115200
- **Data bits**: 8
- **Stop bits**: 1
- **Parity**: None

## 4. Quy Trình Test

### 4.1 Test Từng Module

#### Test LCD1602 + STM32
```c
// Trong main.c, test LCD:
if (LCD_Init() == HAL_OK) {
    LCD_DisplayStartup();
    HAL_Delay(2000);
    LCD_DisplayDamageResult("SCRATCHES", 0.85);
}
```

#### Test UART STM32 ↔ ESP32
- STM32 gửi: `GET_RESULT\r\n`
- ESP32 trả về: JSON response

#### Test ESP32 ↔ Flask Server
- ESP32 upload ảnh lên `/upload`
- ESP32 lấy kết quả từ `/latest_result`

### 4.2 Test End-to-End

1. **Bước 1**: Khởi động Flask Server
   ```bash
   cd app
   python streamlit_app.py
   ```

2. **Bước 2**: Power on STM32 + LCD
   - LCD hiển thị: "CAR DAMAGE AI" → "SYSTEM READY" → "CONNECTING..."

3. **Bước 3**: Power on ESP32-CAM
   - ESP32 kết nối WiFi
   - Gửi "SYSTEM_READY" đến STM32

4. **Bước 4**: Test chụp ảnh
   - ESP32 chụp ảnh mỗi 2 giây
   - Gửi lên Flask server
   - Flask phân tích bằng YOLO11n

5. **Bước 5**: Xem kết quả trên LCD
   - STM32 request kết quả mỗi 3 giây
   - ESP32 forward JSON từ Flask
   - STM32 parse và hiển thị lên LCD

## 5. Giao Diện LCD Hiển Thị

### 5.1 Các Trạng Thái Hiển Thị

#### Khởi động
```
 CAR DAMAGE AI 
 SYSTEM READY  
```

#### Đang kết nối
```
 CONNECTING... 
 PLEASE WAIT  
```

#### Không phát hiện hư hỏng
```
   NO DAMAGE   
  VEHICLE OK   
```

#### Phát hiện hư hỏng
```
DAMAGE DETECTED
SCRATCHES: 85% 
```

#### Lỗi hệ thống
```
    ERROR     
SYSTEM FAULT  
```

### 5.2 Các Loại Hư Hỏng Được Hiển Thị
- `SCRATCHES`: Trầy xước
- `DENTS`: Móp méo  
- `CRACKS`: Nứt vỡ
- `RUST`: Gỉ sét
- `MISSING`: Thiếu bộ phận
- `BROKEN`: Đèn vỡ
- `FLAT`: Lốp xẹp
- `BUMPER`: Cản hỏng

## 6. Troubleshooting

### 6.1 LCD Không Hiển Thị
- Kiểm tra kết nối I2C (SDA, SCL)
- Kiểm tra nguồn 5V/3.3V
- Test địa chỉ I2C bằng I2C scanner
- Kiểm tra pull-up resistor (thường có sẵn trên module)

### 6.2 UART Không Hoạt Động  
- Kiểm tra kết nối TX ↔ RX (chéo nhau)
- Đảm bảo GND chung
- Kiểm tra baud rate (115200)
- Dùng oscilloscope/logic analyzer để debug

### 6.3 ESP32 Không Kết Nối WiFi
- Kiểm tra SSID và password
- Đảm bảo ESP32 trong phạm vi WiFi
- Reset ESP32 và thử lại
- Kiểm tra nguồn 5V ổn định

### 6.4 Flask Server Không Phản Hồi
- Kiểm tra Flask server đang chạy trên port 5000
- Kiểm tra firewall Windows
- Ping test IP 192.168.1.24
- Kiểm tra endpoint bằng browser: `http://192.168.1.24:5000/latest_result`

## 7. Debug và Monitor

### 7.1 Serial Monitor ESP32
- Baud rate: 115200
- Monitor hoạt động của ESP32:
  ```
  WiFi connected: 192.168.1.27
  Photo taken, size: 12543 bytes
  Upload response [200]: {"status": "success", "result": "Safe"}
  New result from server: {"status":"no_damage"...}
  Sent to STM32: {"status":"no_damage"...}
  ```

### 7.2 Debug STM32
- Dùng ST-Link Utility hoặc STM32CubeIDE
- Breakpoint tại UART callback
- Monitor biến `g_latest_result`

### 7.3 Flask Server Logs
```bash
# Trong terminal chạy Flask:
[14:30:25] 📸 Image upload request received from 192.168.1.27
✅ [14:30:25] Image processed successfully!
   📏 Image size: (800, 600)
   🎯 Detections: 0
   📊 Response: Safe
```

## 8. Mở Rộng Hệ Thống

### 8.1 Thêm Buzzer Cảnh Báo
- Kết nối buzzer vào STM32 (ví dụ PA8)
- Kêu khi phát hiện hư hỏng

### 8.2 Thêm LED Status
- LED xanh: Hệ thống OK
- LED đỏ: Phát hiện hư hỏng  
- LED vàng: Đang xử lý

### 8.3 Lưu Trữ Kết Quả
- Thêm SD Card module
- Lưu log kết quả phân tích
- Timestamp với RTC module

### 8.4 Giao Tiếp Không Dây
- Thay UART bằng ESP-NOW
- Hoặc dùng LoRa cho khoảng cách xa

## 9. Sơ Đồ Kết Nối Chi Tiết

```
ESP32-CAM AI-Thinker:
┌─────────────────┐
│  5V    GND      │
│  GPIO1 GPIO3    │──> UART to STM32
│  [Camera]       │
│  [WiFi Antenna] │
└─────────────────┘

STM32F103C8T6 Blue Pill:
┌─────────────────┐
│ PA9  PA10       │──> UART to ESP32
│ PB6  PB7        │──> I2C to LCD  
│ PC13 (LED)      │──> Status LED
│ 3V3  GND        │
└─────────────────┘

LCD1602 + I2C Module:
┌─────────────────┐
│ VCC  GND        │
│ SDA  SCL        │──> I2C to STM32
│ [16x2 Display]  │
└─────────────────┘
```

## 10. Checklist Hoàn Thành

- [ ] ✅ Tạo thư viện LCD1602 cho STM32
- [ ] ✅ Cập nhật Flask server với endpoint `/latest_result`  
- [ ] ✅ Cập nhật code STM32 main.c với UART + LCD
- [ ] ✅ Tạo ESP32 bridge code
- [ ] 🔄 Test kết nối phần cứng
- [ ] 🔄 Test giao tiếp UART ESP32 ↔ STM32
- [ ] 🔄 Test hiển thị LCD1602
- [ ] 🔄 Test end-to-end system

**Tiếp theo**: Kết nối phần cứng theo sơ đồ và test từng bước một cách có hệ thống.