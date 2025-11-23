# PCF8574 LCD1602 Hardware Testing Guide

## Hardware Information
- **Module**: PCF8574 I2C LCD expander
- **Part Number**: PCF8574ABW958 (as identified by user)
- **Default I2C Address**: 0x20 (updated in code)
- **LCD**: LCD1602A (16x2 character display)

## PCF8574 Pin Mapping for LCD1602
```
PCF8574 Pin | LCD Pin | Function
------------|---------|----------
P0          | RS      | Register Select
P1          | RW      | Read/Write (tied to GND)
P2          | EN      | Enable
P3          | A       | Backlight Anode (+)
P4          | D4      | Data bit 4
P5          | D5      | Data bit 5
P6          | D6      | Data bit 6
P7          | D7      | Data bit 7
```

## I2C Address Configuration
The PCF8574 uses different addresses than standard LCD modules:

### PCF8574 Address Range: 0x20 - 0x27
- A2=0, A1=0, A0=0 → 0x20 ✓ (Current setting)
- A2=0, A1=0, A0=1 → 0x21
- A2=0, A1=1, A0=0 → 0x22
- A2=0, A1=1, A0=1 → 0x23
- A2=1, A0=0, A0=0 → 0x24
- A2=1, A1=0, A0=1 → 0x25
- A2=1, A1=1, A0=0 → 0x26
- A2=1, A1=1, A0=1 → 0x27

### PCF8574A Address Range: 0x38 - 0x3F
- Different chip variant (less common)

## Hardware Connection Check
### STM32F103C8T6 to PCF8574:
```
STM32 Pin | PCF8574 Pin | Function
----------|-------------|----------
PB6       | SCL         | I2C Clock
PB7       | SDA         | I2C Data
3.3V      | VCC         | Power
GND       | GND         | Ground
```

### Power Requirements:
- **PCF8574**: 3.3V or 5V compatible
- **LCD1602A**: Typically 5V, but some work at 3.3V
- **Backlight**: Usually requires 5V for full brightness

## Test Procedures

### 1. I2C Scanner Test
```c
// In main.c, call this function:
I2C_ScanDevices();
```
**Expected Result**: Should find device at address 0x20

### 2. LCD Address Test
```c
// In main.c, call this function:
uint8_t found_addr = LCD_FindWorkingAddress();
LCD_SignalFoundAddress(found_addr);
```
**LED Signal Interpretation**:
- Long blink (1 sec) → Test starting
- If found: Blink pattern showing hex address
  - First nibble, pause, second nibble
  - For 0x20: 2 blinks, pause, 0 blinks (no blinks for 0)
- If not found: 10 fast blinks

### 3. Basic LCD Test
```c
// In main.c, call this function:
LCD_TestBasicOperations();
```

## Troubleshooting Guide

### Issue: Fast LED Blinking (Device Not Found)
**Cause**: I2C communication failure
**Solutions**:
1. Check wiring connections
2. Verify power supply (5V for LCD backlight)
3. Test different I2C addresses
4. Check pull-up resistors on SDA/SCL

### Issue: LCD Backlight On, No Characters
**Cause**: Command/data communication issue
**Solutions**:
1. Verify PCF8574 pin mapping
2. Check LCD contrast adjustment (V0 potentiometer)
3. Test with different initialization sequence
4. Verify 4-bit communication timing

### Issue: Corrupted Characters
**Cause**: Timing or electrical issues
**Solutions**:
1. Increase delays in LCD_WriteNibble()
2. Check power supply stability
3. Add decoupling capacitors
4. Verify ground connections

## Code Updates Made

### 1. I2C Address Change
```c
// In lcd1602_i2c.h
#define LCD_I2C_ADDRESS 0x20  // Changed from 0x3F to 0x20
```

### 2. PCF8574-Specific Initialization
- Updated timing delays for PCF8574
- Added proper backlight control
- Enhanced error signaling via LED

### 3. Address Testing
- Priority testing of PCF8574 addresses (0x20-0x27)
- Automatic address detection
- LED feedback for found addresses

## Test Sequence

1. **Upload Code**: Upload updated STM32 code with 0x20 address
2. **Power Check**: Verify 5V power to LCD module
3. **Connection Check**: Verify PB6/PB7 I2C connections
4. **I2C Scan**: Should detect device at 0x20
5. **LCD Init**: Should show successful initialization
6. **Character Test**: Test character display

## Expected LED Patterns

### Successful Operation:
1. Brief power-on blink
2. Long blink (test start)
3. 2 blinks, pause, 0 blinks (address 0x20 found)
4. Long blink (successful LCD init)
5. Steady off (normal operation)

### Failed Operation:
1. Brief power-on blink
2. Long blink (test start)
3. 10 fast blinks (device not found)
4. OR 3 fast blinks (LCD init failed)

## Next Steps

After uploading the updated code:
1. Monitor LED patterns during boot
2. Check if backlight turns on
3. Look for any character display
4. Report the LED blinking pattern observed

## Hardware Verification Checklist

- [ ] PCF8574 module powered with 5V
- [ ] SDA connected to PB7
- [ ] SCL connected to PB6
- [ ] Common ground between STM32 and PCF8574
- [ ] LCD1602A properly seated on PCF8574
- [ ] Contrast potentiometer adjusted (if present)
- [ ] Address jumpers set correctly (usually default 0x20)