#!/usr/bin/env python3
"""
STM32 UART Communication Debug Tool
Giúp debug communication giữa ESP32 và STM32

Sử dụng:
1. Kết nối USB-Serial converter với STM32 UART pins
2. Chạy script này để monitor communication
3. Gửi test commands để kiểm tra STM32 response

Author: Car Damage AI System
Date: 2025-11-26
"""

import serial
import time
import json
import threading
from datetime import datetime

class STM32UARTDebugger:
    def __init__(self, port="COM3", baud=115200):
        """
        Khởi tạo UART debugger
        
        Args:
            port: COM port của USB-Serial converter
            baud: Baud rate (mặc định 115200)
        """
        self.port = port
        self.baud = baud
        self.serial_conn = None
        self.running = False
        
    def connect(self):
        """Kết nối UART"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"✅ Connected to {self.port} at {self.baud} baud")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
            
    def disconnect(self):
        """Ngắt kết nối UART"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("📴 UART disconnected")
            
    def send_command(self, command):
        """Gửi command đến STM32"""
        if self.serial_conn and self.serial_conn.is_open:
            cmd = f"{command}\r\n"
            self.serial_conn.write(cmd.encode())
            print(f"📤 Sent: {command}")
            
    def send_json_response(self, damage_type="scratch", confidence=48.52):
        """Gửi JSON response giống ESP32"""
        json_data = {
            "confidence": confidence,
            "damage_type": damage_type,
            "display_line1": "DAMAGE DETECTED ",
            "display_line2": f"{damage_type.upper()}: {confidence:.0f}%    ",
            "status": "damage_detected",
            "timestamp": datetime.now().isoformat()
        }
        
        json_str = json.dumps(json_data)
        self.send_command(json_str)
        
    def send_no_damage_response(self, confidence=95.0):
        """Gửi response không có damage"""
        json_data = {
            "confidence": confidence,
            "damage_type": "no_damage",
            "display_line1": "  NO DAMAGE     ",
            "display_line2": f"  CONF: {confidence:.0f}%   ",
            "status": "no_damage",
            "timestamp": datetime.now().isoformat()
        }
        
        json_str = json.dumps(json_data)
        self.send_command(json_str)
        
    def monitor_responses(self):
        """Monitor responses từ STM32"""
        print("👀 Monitoring STM32 responses (Ctrl+C to stop)...")
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.readline().decode().strip()
                    if data:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"📥 [{timestamp}] STM32: {data}")
                time.sleep(0.1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                break
                
    def interactive_mode(self):
        """Interactive mode để test communication"""
        print("\n🔧 STM32 UART Interactive Debug Mode")
        print("Commands:")
        print("  1 - Send scratch damage (48%)")
        print("  2 - Send dent damage (75%)")
        print("  3 - Send no damage (95%)")
        print("  4 - Send GET_RESULT command")
        print("  5 - Send custom JSON")
        print("  q - Quit")
        print("-" * 50)
        
        while True:
            try:
                cmd = input("\n> Enter command: ").strip()
                
                if cmd == 'q':
                    break
                elif cmd == '1':
                    self.send_json_response("scratch", 48.52)
                elif cmd == '2':
                    self.send_json_response("dent", 75.3)
                elif cmd == '3':
                    self.send_no_damage_response(95.0)
                elif cmd == '4':
                    self.send_command("GET_RESULT")
                elif cmd == '5':
                    custom_json = input("Enter JSON: ")
                    try:
                        # Validate JSON
                        json.loads(custom_json)
                        self.send_command(custom_json)
                    except json.JSONDecodeError:
                        print("❌ Invalid JSON format")
                else:
                    print("❓ Unknown command")
                    
            except KeyboardInterrupt:
                break
                
    def run_debug_session(self):
        """Chạy debug session hoàn chỉnh"""
        if not self.connect():
            return
            
        self.running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_responses)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Interactive mode
            self.interactive_mode()
        finally:
            self.running = False
            self.disconnect()

def main():
    print("🚗 STM32 Car Damage UART Debugger")
    print("=" * 40)
    
    # Detect available ports
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        print("❌ No serial ports found")
        return
        
    print("Available COM ports:")
    for i, port in enumerate(ports):
        print(f"  {i+1}. {port.device} - {port.description}")
        
    try:
        port_idx = int(input(f"\nSelect port (1-{len(ports)}): ")) - 1
        if 0 <= port_idx < len(ports):
            selected_port = ports[port_idx].device
        else:
            print("❌ Invalid selection")
            return
    except ValueError:
        print("❌ Invalid input")
        return
        
    # Create debugger and run
    debugger = STM32UARTDebugger(port=selected_port, baud=115200)
    debugger.run_debug_session()

if __name__ == "__main__":
    main()