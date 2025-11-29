#!/usr/bin/env python3
"""
STM32-ESP32 Communication Debugger
==================================

This tool helps debug the communication between STM32 and ESP32
for the car damage detection system.

Features:
- Monitor UART communication
- Test JSON parsing
- Simulate STM32 responses
- Debug LCD display formatting

Author: AI Assistant
Date: November 2025
"""

import json
import time
import re

def test_json_parsing():
    """Test various JSON responses that ESP32 might send"""
    
    print("🧪 Testing JSON Response Parsing")
    print("=" * 50)
    
    # Test cases from Flask server
    test_responses = [
        # Success case with damage
        {
            "status": "success",
            "damage_detected": True,
            "damage_type": "Scratch", 
            "confidence": 85.2,
            "severity": 1,
            "timestamp": "2025-11-26T10:30:00"
        },
        
        # No damage case
        {
            "status": "success",
            "damage_detected": False,
            "damage_type": "No Damage",
            "confidence": 95.0,
            "severity": 0,
            "timestamp": "2025-11-26T10:30:00"
        },
        
        # Multiple damage types
        {
            "status": "success",
            "damage_detected": True,
            "damage_type": "Dent",
            "confidence": 78.5,
            "severity": 3,
            "timestamp": "2025-11-26T10:30:00"
        },
        
        # Long damage type name
        {
            "status": "success", 
            "damage_detected": True,
            "damage_type": "Rust Corrosion",
            "confidence": 92.1,
            "severity": 2,
            "timestamp": "2025-11-26T10:30:00"
        },
        
        # Error case
        {
            "status": "error",
            "message": "Analysis failed"
        }
    ]
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n📋 Test Case {i}:")
        print(f"JSON: {json.dumps(response)}")
        
        # Simulate STM32 parsing
        lcd_display = simulate_stm32_parsing(json.dumps(response))
        
        print(f"LCD Display:")
        print("┌────────────────┐")
        print(f"│{lcd_display['line1']}│")
        print(f"│{lcd_display['line2']}│")
        print("└────────────────┘")

def simulate_stm32_parsing(json_str):
    """Simulate how STM32 parses JSON and formats for LCD"""
    
    try:
        data = json.loads(json_str)
        
        # Extract values like STM32 would
        status = data.get('status', 'unknown')
        damage_type = data.get('damage_type', 'UNKNOWN')
        confidence = data.get('confidence', 0.0)
        severity = data.get('severity', 0)
        
        # Format for LCD (16 characters per line)
        if status == 'success':
            # Line 1: Damage Type (formatted like STM32)
            formatted_type = damage_type.upper().replace('_', ' ')
            
            if len(formatted_type) > 16:
                line1 = formatted_type[:13] + "..."
            else:
                # Center text
                padding = (16 - len(formatted_type)) // 2
                line1 = ' ' * padding + formatted_type
                line1 = line1.ljust(16)
            
            # Line 2: Confidence (formatted like STM32)
            if 'NO DAMAGE' in formatted_type or 'SYSTEM READY' in formatted_type:
                line2 = f"   CONF: {confidence:.0f}%   "
            else:
                if severity > 0:
                    line2 = f"{confidence:.0f}% | SEV: {severity}"
                else:
                    line2 = f"  CONF: {confidence:.0f}%  "
            
            line2 = line2.ljust(16)[:16]
            
        else:
            line1 = "  SYSTEM ERROR  "
            line2 = " CHECK ESP32-CAM"
        
        return {
            'line1': line1[:16],
            'line2': line2[:16]
        }
        
    except json.JSONDecodeError:
        # Not JSON, treat as status message
        if 'SYSTEM_READY' in json_str or 'WIFI_CONNECTED' in json_str:
            return {
                'line1': '  SYSTEM READY  ',
                'line2': '   CONF: 100%   '
            }
        else:
            return {
                'line1': '  COMM ERROR    ',
                'line2': ' INVALID DATA  '
            }

def simulate_uart_communication():
    """Simulate UART communication between STM32 and ESP32"""
    
    print("\n🔌 UART Communication Simulation")
    print("=" * 50)
    
    # Simulate ESP32 responses
    esp32_responses = [
        "SYSTEM_READY",
        "WIFI_CONNECTED", 
        "CAMERA_READY",
        '{"status":"success","damage_detected":true,"damage_type":"Scratch","confidence":85.2,"severity":1}',
        '{"status":"success","damage_detected":false,"damage_type":"No Damage","confidence":95.0,"severity":0}',
        '{"status":"error","message":"Camera failed"}',
        "HEARTBEAT:WIFI_OK,CAM_OK,SERVER_OK"
    ]
    
    for response in esp32_responses:
        print(f"\n📡 ESP32 → STM32: {response}")
        
        # Show how STM32 would display this
        lcd_display = simulate_stm32_parsing(response)
        
        print(f"STM32 LCD Display:")
        print("┌────────────────┐")
        print(f"│{lcd_display['line1']}│")
        print(f"│{lcd_display['line2']}│")
        print("└────────────────┘")
        
        time.sleep(1)

def test_lcd_formatting():
    """Test LCD display formatting for various damage types"""
    
    print("\n📺 LCD Display Formatting Tests")
    print("=" * 50)
    
    damage_types = [
        ("Scratch", 85.2, 1),
        ("Dent", 78.5, 3), 
        ("Crack", 92.1, 2),
        ("Rust Corrosion", 67.8, 2),
        ("Paint Damage", 73.4, 1),
        ("No Damage", 95.0, 0),
        ("Broken Light", 89.3, 4),
        ("Very Long Damage Name", 56.7, 2)
    ]
    
    for damage_type, confidence, severity in damage_types:
        json_response = {
            "status": "success",
            "damage_detected": damage_type != "No Damage",
            "damage_type": damage_type,
            "confidence": confidence,
            "severity": severity
        }
        
        lcd_display = simulate_stm32_parsing(json.dumps(json_response))
        
        print(f"\nDamage: {damage_type} ({confidence}%, Sev:{severity})")
        print("┌────────────────┐")
        print(f"│{lcd_display['line1']}│")
        print(f"│{lcd_display['line2']}│")
        print("└────────────────┘")

def generate_debug_commands():
    """Generate debug commands for testing"""
    
    print("\n🛠️  Debug Commands for Testing")
    print("=" * 50)
    
    print("STM32 → ESP32 Commands:")
    commands = [
        ("GET_RESULT", "Request latest analysis result"),
        ("GET_STATUS", "Request system status"),
        ("TAKE_PHOTO", "Force camera capture")
    ]
    
    for cmd, desc in commands:
        print(f"  {cmd:<12} - {desc}")
    
    print("\nESP32 → STM32 Responses:")
    responses = [
        ("JSON Result", '{"status":"success","damage_type":"Scratch","confidence":85.2,"severity":1}'),
        ("System Ready", "SYSTEM_READY"),
        ("WiFi Status", "WIFI_CONNECTED"),
        ("Heartbeat", "HEARTBEAT:WIFI_OK,CAM_OK,SERVER_OK"),
        ("Error", '{"status":"error","message":"Camera failed"}')
    ]
    
    for name, response in responses:
        print(f"  {name:<12} - {response}")

def main():
    """Main function"""
    
    print("🚗 STM32-ESP32 Communication Debugger")
    print("=" * 60)
    print("This tool helps debug the car damage detection system")
    print("communication between STM32 and ESP32-CAM modules.")
    print("=" * 60)
    
    while True:
        print("\n📋 Select test to run:")
        print("1. Test JSON parsing")
        print("2. Simulate UART communication")
        print("3. Test LCD formatting")
        print("4. Generate debug commands")
        print("5. Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                test_json_parsing()
            elif choice == '2':
                simulate_uart_communication()
            elif choice == '3':
                test_lcd_formatting()
            elif choice == '4':
                generate_debug_commands()
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()