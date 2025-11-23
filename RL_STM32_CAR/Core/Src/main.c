/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
// LCD Configuration - Found working address
#define LCD_I2C_ADDR 0x27
uint8_t lcd_initialized = 0;
uint32_t test_counter = 0;
uint32_t last_test_time = 0;
uint8_t current_test_phase = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void LCD_Send_Nibble(uint8_t nibble, uint8_t rs);
void LCD_Send_Byte(uint8_t byte, uint8_t rs);
void LCD_Init(void);
void LCD_Clear(void);
void LCD_SetCursor(uint8_t col, uint8_t row);
void LCD_Print(char* str);
void LCD_Test_Basic(void);
void LCD_Test_Characters(void);
void LCD_Test_Backlight(void);
void LCD_Test_Scrolling(void);
void LCD_Display_System_Info(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_TIM2_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
  
  // Power-on signal - 1 long blink
  HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_SET);
  HAL_Delay(1000);
  HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_RESET);
  HAL_Delay(500);
  
  // Wait for LCD power stabilization
  HAL_Delay(1000);
  
  // Initialize LCD with known working address
  if (HAL_I2C_IsDeviceReady(&hi2c1, LCD_I2C_ADDR << 1, 3, 1000) == HAL_OK) {
    LCD_Init();
    lcd_initialized = 1;
    
    // Success - 3 quick blinks
    for(int i = 0; i < 3; i++) {
      HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_SET);
      HAL_Delay(200);
      HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_RESET);
      HAL_Delay(200);
    }
    
    // Display welcome message
    LCD_Clear();
    LCD_SetCursor(0, 0);
    LCD_Print("LCD1602 READY!");
    LCD_SetCursor(0, 1);
    LCD_Print("ADDR: 0x27");
    HAL_Delay(2000);
    
  } else {
    // Failed - 5 fast blinks
    for(int i = 0; i < 5; i++) {
      HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_SET);
      HAL_Delay(100);
      HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_RESET);
      HAL_Delay(100);
    }
    lcd_initialized = 0;
  }
  
  last_test_time = HAL_GetTick();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    uint32_t current_time = HAL_GetTick();
    static uint32_t led_timer = 0;
    
    if (lcd_initialized) {
      // LCD working - steady LED heartbeat
      if ((current_time - led_timer) >= 1000) {
        led_timer = current_time;
        HAL_GPIO_TogglePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin);
      }
      
      // Run test sequences every 6 seconds
      if ((current_time - last_test_time) >= 6000) {
        last_test_time = current_time;
        
        switch(current_test_phase) {
          case 0:
            LCD_Test_Basic();
            break;
          case 1:
            LCD_Test_Characters();
            break;
          case 2:
            LCD_Test_Backlight();
            break;
          case 3:
            LCD_Test_Scrolling();
            break;
          case 4:
            LCD_Display_System_Info();
            break;
          default:
            current_test_phase = 0;
            continue;
        }
        current_test_phase++;
        if (current_test_phase > 4) current_test_phase = 0;
      }
      
    } else {
      // LCD not working - fast blink and retry
      if ((current_time - led_timer) >= 200) {
        led_timer = current_time;
        HAL_GPIO_TogglePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin);
      }
      
      // Retry LCD every 5 seconds
      if ((current_time - last_test_time) >= 5000) {
        last_test_time = current_time;
        if (HAL_I2C_IsDeviceReady(&hi2c1, LCD_I2C_ADDR << 1, 3, 1000) == HAL_OK) {
          LCD_Init();
          lcd_initialized = 1;
        }
      }
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/**
 * @brief Send nibble to LCD via PCF8574 (optimized for 0x27)
 * @param nibble: 4-bit data
 * @param rs: Register select (0=command, 1=data)
 */
void LCD_Send_Nibble(uint8_t nibble, uint8_t rs) {
  uint8_t data = nibble & 0xF0;
  if (rs) data |= 0x01;  // RS bit
  data |= 0x08;          // Backlight bit
  
  // Send with E high
  HAL_I2C_Master_Transmit(&hi2c1, LCD_I2C_ADDR << 1, &data, 1, 100);
  data |= 0x04;  // E high
  HAL_I2C_Master_Transmit(&hi2c1, LCD_I2C_ADDR << 1, &data, 1, 100);
  HAL_Delay(1);
  
  // Send with E low
  data &= ~0x04; // E low
  HAL_I2C_Master_Transmit(&hi2c1, LCD_I2C_ADDR << 1, &data, 1, 100);
  HAL_Delay(1);
}

/**
 * @brief Send byte to LCD
 * @param byte: 8-bit data
 * @param rs: Register select (0=command, 1=data)
 */
void LCD_Send_Byte(uint8_t byte, uint8_t rs) {
  LCD_Send_Nibble(byte & 0xF0, rs);      // Upper nibble
  LCD_Send_Nibble((byte << 4) & 0xF0, rs); // Lower nibble
}

/**
 * @brief Initialize LCD
 */
void LCD_Init(void) {
  HAL_Delay(50);
  
  // 8-bit mode initialization sequence
  LCD_Send_Nibble(0x30, 0);
  HAL_Delay(5);
  LCD_Send_Nibble(0x30, 0);
  HAL_Delay(1);
  LCD_Send_Nibble(0x30, 0);
  HAL_Delay(1);
  
  // Switch to 4-bit mode
  LCD_Send_Nibble(0x20, 0);
  HAL_Delay(1);
  
  // Function set: 4-bit, 2 lines, 5x8 dots
  LCD_Send_Byte(0x28, 0);
  HAL_Delay(1);
  
  // Display control: display on, cursor off, blink off
  LCD_Send_Byte(0x0C, 0);
  HAL_Delay(1);
  
  // Clear display
  LCD_Send_Byte(0x01, 0);
  HAL_Delay(2);
  
  // Entry mode: increment, no shift
  LCD_Send_Byte(0x06, 0);
  HAL_Delay(1);
}

/**
 * @brief Clear LCD display
 */
void LCD_Clear(void) {
  LCD_Send_Byte(0x01, 0);
  HAL_Delay(2);
}

/**
 * @brief Set cursor position
 * @param col: Column (0-15)
 * @param row: Row (0-1)
 */
void LCD_SetCursor(uint8_t col, uint8_t row) {
  uint8_t address = (row == 0) ? 0x80 : 0xC0;
  address += col;
  LCD_Send_Byte(address, 0);
  HAL_Delay(1);
}

/**
 * @brief Print string to LCD
 * @param str: String to print
 */
void LCD_Print(char* str) {
  while (*str) {
    LCD_Send_Byte(*str++, 1);
    HAL_Delay(1);
  }
}

/**
 * @brief Test basic LCD functions
 */
void LCD_Test_Basic(void) {
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("** BASIC TEST **");
  HAL_Delay(1500);
  
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("Hello World!");
  LCD_SetCursor(0, 1);
  LCD_Print("LCD1602 Works!");
  HAL_Delay(3000);
  
  // Counter test
  for (int i = 0; i < 10; i++) {
    LCD_Clear();
    LCD_SetCursor(0, 0);
    LCD_Print("Counter Test:");
    LCD_SetCursor(0, 1);
    char num_str[16];
    sprintf(num_str, "Count: %d", i);
    LCD_Print(num_str);
    HAL_Delay(500);
  }
}

/**
 * @brief Test character display
 */
void LCD_Test_Characters(void) {
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("CHARACTER TEST");
  HAL_Delay(1500);
  
  // Numbers and letters
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("0123456789ABCDEF");
  LCD_SetCursor(0, 1);
  LCD_Print("!@#$%^&*()_+-=<>");
  HAL_Delay(3000);
  
  // ASCII characters
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("ASCII: ");
  for (char c = 65; c <= 90; c++) {
    if (c > 65) {
      LCD_SetCursor((c-65) % 16, (c-65) / 16);
    } else {
      LCD_SetCursor(7, 0);
    }
    char temp[2] = {c, '\0'};
    LCD_Print(temp);
    HAL_Delay(200);
  }
  HAL_Delay(1000);
}

/**
 * @brief Test backlight control
 */
void LCD_Test_Backlight(void) {
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("BACKLIGHT TEST");
  LCD_SetCursor(0, 1);
  LCD_Print("Watch the light!");
  HAL_Delay(2000);
  
  // Toggle backlight
  for (int i = 0; i < 5; i++) {
    // Turn backlight off
    uint8_t data = 0x00; // No backlight
    HAL_I2C_Master_Transmit(&hi2c1, LCD_I2C_ADDR << 1, &data, 1, 100);
    HAL_Delay(800);
    
    // Turn backlight on
    data = 0x08; // Backlight on
    HAL_I2C_Master_Transmit(&hi2c1, LCD_I2C_ADDR << 1, &data, 1, 100);
    HAL_Delay(800);
  }
}

/**
 * @brief Test scrolling text
 */
void LCD_Test_Scrolling(void) {
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("SCROLL TEST");
  HAL_Delay(1500);
  
  char *scroll_text = "    This is a long scrolling text message for LCD1602 display testing!    ";
  int text_len = strlen(scroll_text);
  
  for (int pos = 0; pos < text_len - 16; pos++) {
    LCD_Clear();
    LCD_SetCursor(0, 0);
    LCD_Print("SCROLLING:...");
    LCD_SetCursor(0, 1);
    
    // Print 16 characters starting from pos
    for (int i = 0; i < 16 && (pos + i) < text_len; i++) {
      char temp[2] = {scroll_text[pos + i], '\0'};
      LCD_Print(temp);
    }
    HAL_Delay(300);
  }
}

/**
 * @brief Display system information
 */
void LCD_Display_System_Info(void) {
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("SYSTEM INFO");
  HAL_Delay(1500);
  
  // Display uptime
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("Uptime:");
  LCD_SetCursor(0, 1);
  uint32_t uptime_sec = HAL_GetTick() / 1000;
  char uptime_str[16];
  sprintf(uptime_str, "%lu seconds", uptime_sec);
  LCD_Print(uptime_str);
  HAL_Delay(2000);
  
  // Display test cycles
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("Test Cycles:");
  LCD_SetCursor(0, 1);
  char cycles_str[16];
  sprintf(cycles_str, "Cycle: %lu", ++test_counter);
  LCD_Print(cycles_str);
  HAL_Delay(2000);
  
  // Display I2C status
  LCD_Clear();
  LCD_SetCursor(0, 0);
  LCD_Print("I2C Status: OK");
  LCD_SetCursor(0, 1);
  LCD_Print("Addr: 0x27 ✓");
  HAL_Delay(2000);
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
