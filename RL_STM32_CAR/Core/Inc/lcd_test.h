/*
 * LCD1602 Test Functions Header
 * 
 * Test functions để kiểm tra LCD hoạt động
 * 
 * Author: STM32 Car Damage Detection System
 * Date: 2025
 */

#ifndef LCD_TEST_H
#define LCD_TEST_H

#include "main.h"

/* Function Prototypes */
uint8_t LCD_ScanI2C(void);
HAL_StatusTypeDef LCD_SimpleTest(void);
HAL_StatusTypeDef LCD_QuickTest(void);
HAL_StatusTypeDef LCD_TestI2C(void);

#endif /* LCD_TEST_H */