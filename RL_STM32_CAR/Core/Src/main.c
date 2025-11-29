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
#include "Timer.h"
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

// Car Damage Detection System
extern timer_Objt Tim_1ms[MaxTIMER];
uint8_t lcd_initialized = 0;
uint32_t last_analysis_time = 0;
uint32_t analysis_interval = 15000; // Analysis every 15 seconds (slower)
uint8_t uart_rx_buffer[256];
uint8_t uart_rx_index = 0;
uint8_t uart_message_ready = 0;

// Damage detection results
typedef struct {
    char damage_type[32];
    float confidence;
    uint8_t severity;
    uint32_t timestamp;
    uint8_t valid;
} DamageResult_t;

DamageResult_t current_damage;
DamageResult_t last_valid_damage; // Keep last valid result for display
DamageResult_t last_damage_detected; // Keep last actual damage (not no_damage)
uint8_t system_status = 0; // 0=init, 1=ready, 2=analyzing, 3=error
uint32_t test_counter = 0; // Counter for test cycles display
uint8_t should_show_analyzing = 0; // Only show analyzing when actually requesting

// Damage detection optimization
uint32_t last_damage_timestamp = 0; // When was last damage detected
uint32_t damage_display_duration = 30000; // Show damage for 30 seconds
float min_confidence_threshold = 35.0; // Minimum confidence for damage detection
uint8_t consecutive_no_damage = 0; // Count consecutive no_damage results
uint8_t no_damage_threshold = 5; // Need 5 consecutive no_damage to override
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
// LCD Functions
void LCD_Send_Nibble(uint8_t nibble, uint8_t rs);
void LCD_Send_Byte(uint8_t byte, uint8_t rs);
void LCD_Init(void);
void LCD_Clear(void);
void LCD_SetCursor(uint8_t col, uint8_t row);
void LCD_Print(char* str);

// Car Damage Detection Functions
void CarDamage_Init(void);
void CarDamage_RequestAnalysis(void);
void CarDamage_ProcessUARTData(void);
void CarDamage_ParseResponse(char* json_response);
void CarDamage_DisplayResults(void);
void CarDamage_DisplayStatus(void);
void CarDamage_SendESP32Command(char* command);
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart);
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
  
  // Initialize timer system
  HAL_TIM_Base_Start_IT(&htim2);
  startTim(&Tim_1ms[0], 1000);
  
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
    
    // Initialize Car Damage Detection System
    CarDamage_Init();
    
    // Start UART receive interrupt
    HAL_UART_Receive_IT(&huart1, &uart_rx_buffer[0], 1);
    
  } else {
    // Failed - 5 fast blinks
    for(int i = 0; i < 5; i++) {
      HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_SET);
      HAL_Delay(100);
      HAL_GPIO_WritePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin, GPIO_PIN_RESET);
      HAL_Delay(100);
    }
    lcd_initialized = 0;
    system_status = 3; // Error
  }
  
  last_analysis_time = HAL_GetTick();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    uint32_t current_time = HAL_GetTick();
    static uint32_t led_timer = 0;
    
    // Handle timer system
    scanTimer();
    
    if (lcd_initialized) {
      // Process UART data from ESP32
      CarDamage_ProcessUARTData();
      
      // LED patterns based on system status
      uint32_t blink_interval = 1000; // Default
      switch(system_status) {
        case 0: blink_interval = 500; break;  // Init - fast blink
        case 1: blink_interval = 2000; break; // Ready - slow blink
        case 2: blink_interval = 250; break;  // Analyzing - very fast
        case 3: blink_interval = 100; break;  // Error - ultra fast
      }
      
      if ((current_time - led_timer) >= blink_interval) {
        led_timer = current_time;
        HAL_GPIO_TogglePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin);
      }
      
      // Smart analysis interval - analyze more frequently if no damage found
      uint32_t dynamic_interval = analysis_interval;
      uint32_t time_since_damage = HAL_GetTick() - last_damage_timestamp;
      
      if (last_damage_timestamp == 0 || time_since_damage > damage_display_duration) {
        // No recent damage - analyze more frequently
        dynamic_interval = 8000; // Every 8 seconds
      } else {
        // Recent damage found - analyze less frequently to avoid false negatives
        dynamic_interval = 20000; // Every 20 seconds
      }
      
      // Request analysis every interval (only if not analyzing)
      if ((current_time - last_analysis_time) >= dynamic_interval && system_status == 1) {
        last_analysis_time = current_time;
        CarDamage_RequestAnalysis();
      }
      
      // Timeout protection for analyzing state - shorter timeout, no error display
      if (system_status == 2 && (current_time - last_analysis_time) > 3000) {
        // Timeout after 3 seconds, go back to ready without showing error
        system_status = 1;
        should_show_analyzing = 0;
        // Don't show timeout error, just continue with last valid result
      }
      
      // Update display every 1 second or when new data arrives
      static uint32_t display_timer = 0;
      static uint32_t last_damage_timestamp = 0;
      static uint32_t analyzing_start_time = 0;
      
      // Track when analyzing started
      if (system_status == 2 && should_show_analyzing && analyzing_start_time == 0) {
        analyzing_start_time = current_time;
      } else if (system_status != 2) {
        analyzing_start_time = 0;
      }
      
      if ((current_time - display_timer) >= 1000 || 
          (current_damage.valid && current_damage.timestamp != last_damage_timestamp)) {
        display_timer = current_time;
        last_damage_timestamp = current_damage.timestamp;
        
        // Priority order for display:
        // 1. Show analyzing only for first 2 seconds and if should_show_analyzing is true
        // 2. Show current valid result
        // 3. Show last valid result 
        // 4. Show status only if no results available
        
        if (system_status == 2 && should_show_analyzing && 
            analyzing_start_time > 0 && (current_time - analyzing_start_time) < 2000) {
          // Show analyzing for only 2 seconds
          // Already displayed in CarDamage_RequestAnalysis, do nothing
        } else {
          // Smart display logic - prioritize damage results
          uint32_t time_since_damage = HAL_GetTick() - last_damage_timestamp;
          
          if (last_damage_detected.valid && time_since_damage <= damage_display_duration) {
            // Show last detected damage if within time window
            memcpy(&current_damage, &last_damage_detected, sizeof(DamageResult_t));
            CarDamage_DisplayResults();
          } else if (last_valid_damage.valid) {
            // Show last valid result (could be no_damage)
            memcpy(&current_damage, &last_valid_damage, sizeof(DamageResult_t));
            CarDamage_DisplayResults();
          } else {
            CarDamage_DisplayStatus();
          }
        }
      }
      
    } else {
      // LCD not working - error blink and retry
      if ((current_time - led_timer) >= 200) {
        led_timer = current_time;
        HAL_GPIO_TogglePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin);
      }
      
      // Retry LCD every 5 seconds
      if ((current_time - last_analysis_time) >= 5000) {
        last_analysis_time = current_time;
        if (HAL_I2C_IsDeviceReady(&hi2c1, LCD_I2C_ADDR << 1, 3, 1000) == HAL_OK) {
          LCD_Init();
          CarDamage_Init();
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

// Car Damage Detection System Functions
void CarDamage_Init(void) {
    // Initialize car damage detection system
    memset(&current_damage, 0, sizeof(DamageResult_t));
    memset(&last_valid_damage, 0, sizeof(DamageResult_t));
    memset(&last_damage_detected, 0, sizeof(DamageResult_t));
    
    // Set default values for last_valid_damage
    strcpy(last_valid_damage.damage_type, "SYSTEM READY");
    last_valid_damage.confidence = 100.0;
    last_valid_damage.severity = 0;
    last_valid_damage.valid = 1;
    last_valid_damage.timestamp = HAL_GetTick();
    
    // Initialize damage detection optimization
    last_damage_timestamp = 0;
    consecutive_no_damage = 0;
    
    memset(uart_rx_buffer, 0, sizeof(uart_rx_buffer));
    uart_rx_index = 0;
    uart_message_ready = 0;
    system_status = 0;
    should_show_analyzing = 0;
    
    LCD_Clear();
    LCD_SetCursor(0, 0);
    LCD_Print("  CAR DAMAGE AI ");
    LCD_SetCursor(0, 1);
    LCD_Print(" INITIALIZING...");
    HAL_Delay(2000);
    
    // Send request to ESP32 to get latest result
    CarDamage_SendESP32Command("GET_RESULT");
    HAL_Delay(500);
    
    system_status = 1; // Ready state
    
    // Display ready status
    LCD_Clear();
    LCD_SetCursor(0, 0);
    LCD_Print("  SYSTEM READY  ");
    LCD_SetCursor(0, 1);
    LCD_Print("  WAITING DATA  ");
}

void CarDamage_RequestAnalysis(void) {
    if (system_status == 1) { // Only if system is ready
        system_status = 2; // Analyzing state
        should_show_analyzing = 1; // Flag to show analyzing
        
        // Only show analyzing for 2 seconds, then keep old result
        LCD_Clear();
        LCD_SetCursor(0, 0);
        LCD_Print("   ANALYZING    ");
        LCD_SetCursor(0, 1);
        LCD_Print(" PLEASE WAIT... ");
        
        // Request latest result from ESP32
        CarDamage_SendESP32Command("GET_RESULT");
        last_analysis_time = HAL_GetTick();
    }
}

void CarDamage_ProcessUARTData(void) {
    if (uart_message_ready) {
        uart_message_ready = 0;
        
        // Process received JSON data
        if (strstr((char*)uart_rx_buffer, "{") != NULL) {
            CarDamage_ParseResponse((char*)uart_rx_buffer);
            
            // Smart damage detection logic
            if (current_damage.valid) {
                // Check if this is actual damage detection (not no_damage)
                if (strstr(current_damage.damage_type, "NO DAMAGE") == NULL && 
                    strstr(current_damage.damage_type, "SYSTEM READY") == NULL &&
                    current_damage.confidence >= min_confidence_threshold) {
                    
                    // This is actual damage with good confidence
                    memcpy(&last_damage_detected, &current_damage, sizeof(DamageResult_t));
                    last_damage_timestamp = HAL_GetTick();
                    consecutive_no_damage = 0; // Reset no damage counter
                    
                    // Update display immediately with damage
                    memcpy(&last_valid_damage, &current_damage, sizeof(DamageResult_t));
                    
                } else if (strstr(current_damage.damage_type, "NO DAMAGE") != NULL) {
                    // This is no damage result
                    consecutive_no_damage++;
                    
                    // Only override damage display if:
                    // 1. We have many consecutive no_damage results AND
                    // 2. Last damage was detected long ago (>30 seconds)
                    uint32_t time_since_damage = HAL_GetTick() - last_damage_timestamp;
                    
                    if (consecutive_no_damage >= no_damage_threshold && 
                        time_since_damage > damage_display_duration) {
                        // Override with no damage after threshold
                        memcpy(&last_valid_damage, &current_damage, sizeof(DamageResult_t));
                    }
                    // Otherwise keep showing the last detected damage
                    
                } else {
                    // Other status messages
                    memcpy(&last_valid_damage, &current_damage, sizeof(DamageResult_t));
                }
            }
            
            system_status = 1; // Back to ready state
            should_show_analyzing = 0; // Stop showing analyzing
            
        } else if (strstr((char*)uart_rx_buffer, "SYSTEM_READY") || 
                   strstr((char*)uart_rx_buffer, "WIFI_CONNECTED") ||
                   strstr((char*)uart_rx_buffer, "CAMERA_READY")) {
            // Handle non-JSON status messages
            CarDamage_ParseResponse((char*)uart_rx_buffer);
            
            if (current_damage.valid) {
                memcpy(&last_valid_damage, &current_damage, sizeof(DamageResult_t));
            }
            
            system_status = 1; // Back to ready state
            should_show_analyzing = 0; // Stop showing analyzing
        }
        
        // Clear buffer and reset index
        memset(uart_rx_buffer, 0, sizeof(uart_rx_buffer));
        uart_rx_index = 0;
        
        // Restart UART interrupt for next message
        HAL_UART_Receive_IT(&huart1, &uart_rx_buffer[0], 1);
    }
}

void CarDamage_ParseResponse(char* json_response) {
    // Initialize default values
    memset(&current_damage, 0, sizeof(DamageResult_t));
    current_damage.valid = 0;
    
    // Check if it's a JSON response
    if (strstr(json_response, "{") == NULL) {
        // Not JSON, treat as simple status message
        if (strstr(json_response, "SYSTEM_READY") || 
            strstr(json_response, "WIFI_CONNECTED") ||
            strstr(json_response, "CAMERA_READY")) {
            strcpy(current_damage.damage_type, "SYSTEM READY");
            current_damage.confidence = 100.0;
            current_damage.severity = 0;
            current_damage.valid = 1;
        }
        return;
    }
    
    // Parse JSON response
    char* status_ptr = strstr(json_response, "\"status\"");
    char* damage_type_ptr = strstr(json_response, "\"damage_type\"");
    char* confidence_ptr = strstr(json_response, "\"confidence\"");
    char* severity_ptr = strstr(json_response, "\"severity\"");
    
    // Check if analysis was successful - handle new format
    if (status_ptr && (strstr(status_ptr, "damage_detected") || strstr(status_ptr, "no_damage") || strstr(status_ptr, "no_data"))) {
        current_damage.valid = 1;
        
        // Extract damage type
        if (damage_type_ptr) {
            char* start = strstr(damage_type_ptr, ":\"");
            if (start) {
                start += 2; // Skip :"
                char* end = strstr(start, "\"");
                if (end && (end - start) < sizeof(current_damage.damage_type) - 1) {
                    strncpy(current_damage.damage_type, start, end - start);
                    current_damage.damage_type[end - start] = '\0';
                }
            }
        }
        
        // Extract confidence (handle decimal format)
        if (confidence_ptr) {
            char* start = strstr(confidence_ptr, ":");
            if (start) {
                start += 1;
                // Skip whitespace and find the number
                while (*start == ' ' || *start == '\t') start++;
                float conf = atof(start);
                
                // Convert to percentage if needed (0.485 -> 48.5%)
                if (conf > 0.0) {
                    current_damage.confidence = (conf < 1.0) ? conf * 100.0 : conf;
                    // Ensure reasonable range
                    if (current_damage.confidence > 100.0) {
                        current_damage.confidence = 100.0;
                    }
                } else {
                    current_damage.confidence = 50.0; // Default if parsing failed
                }
            }
        }
        
        // Extract severity
        if (severity_ptr) {
            char* start = strstr(severity_ptr, ":");
            if (start) {
                start += 1;
                current_damage.severity = (uint8_t)atoi(start);
            }
        }
        
        // Set default values if not found
        if (strlen(current_damage.damage_type) == 0) {
            strcpy(current_damage.damage_type, "NO DAMAGE");
        }
        if (current_damage.confidence <= 0.0 || current_damage.confidence > 100.0) {
            current_damage.confidence = 95.0; // Default confidence
        }
        
        current_damage.timestamp = HAL_GetTick();
        
    } else {
        // Error or unknown response
        strcpy(current_damage.damage_type, "COMM ERROR");
        current_damage.confidence = 0.0;
        current_damage.severity = 0;
        current_damage.valid = 0;
    }
}

void CarDamage_DisplayResults(void) {
    if (current_damage.valid) {
        LCD_Clear();
        
        // Line 1: Damage Type (16 characters max)
        LCD_SetCursor(0, 0);
        char line1[17] = {0};
        
        // Format damage type for display (uppercase, replace underscores)
        char formatted_type[32];
        strcpy(formatted_type, current_damage.damage_type);
        
        // Convert to uppercase and replace underscores with spaces
        for (int i = 0; formatted_type[i]; i++) {
            if (formatted_type[i] == '_') formatted_type[i] = ' ';
            if (formatted_type[i] >= 'a' && formatted_type[i] <= 'z') {
                formatted_type[i] = formatted_type[i] - 32; // Convert to uppercase
            }
        }
        
        // Truncate if too long
        if (strlen(formatted_type) > 16) {
            strncpy(line1, formatted_type, 13);
            strcat(line1, "...");
        } else {
            // Center the text if shorter
            int padding = (16 - strlen(formatted_type)) / 2;
            for (int i = 0; i < padding; i++) line1[i] = ' ';
            strcpy(line1 + padding, formatted_type);
        }
        LCD_Print(line1);
        
        // Line 2: Confidence (and severity if damage detected)
        LCD_SetCursor(0, 1);
        char line2[17] = {0};
        
        // Ensure confidence has a valid value
        float display_confidence = current_damage.confidence;
        if (display_confidence <= 0.0) {
            display_confidence = 95.0; // Default fallback
        }
        
        if (strstr(formatted_type, "NO DAMAGE") || strstr(formatted_type, "SYSTEM READY")) {
            // No damage - show confidence and analysis count
            sprintf(line2, "CONF:%.0f%% N:%d", display_confidence, consecutive_no_damage);
        } else {
            // Damage detected - show confidence and time info
            uint32_t time_since_damage = HAL_GetTick() - last_damage_timestamp;
            uint32_t seconds_ago = time_since_damage / 1000;
            
            if (seconds_ago < 60) {
                sprintf(line2, "%.0f%% | %lus ago", display_confidence, seconds_ago);
            } else {
                sprintf(line2, "%.0f%% | %lum ago", display_confidence, seconds_ago / 60);
            }
        }
        
        // Ensure line2 is exactly 16 characters and pad if needed
        int len = strlen(line2);
        while (len < 16) {
            line2[len++] = ' ';
        }
        line2[16] = '\0';
        LCD_Print(line2);
        
    } else {
        // System error display
        LCD_Clear();
        LCD_SetCursor(0, 0);
        LCD_Print("  SYSTEM ERROR  ");
        LCD_SetCursor(0, 1);
        LCD_Print(" CHECK ESP32-CAM");
        system_status = 3; // Error state
    }
}

void CarDamage_DisplayStatus(void) {
    // Display system status on LCD
    LCD_Clear();
    LCD_SetCursor(0, 0);
    
    switch(system_status) {
        case 0:
            LCD_Print("INITIALIZING...");
            break;
        case 1:
            LCD_Print("SYSTEM READY");
            LCD_SetCursor(0, 1);
            uint32_t uptime = HAL_GetTick() / 1000;
            char uptime_str[17];
            sprintf(uptime_str, "Uptime: %lus", uptime);
            LCD_Print(uptime_str);
            break;
        case 2:
            LCD_Print("ANALYZING...");
            LCD_SetCursor(0, 1);
            LCD_Print("Please wait...");
            break;
        case 3:
            LCD_Print("SYSTEM ERROR");
            LCD_SetCursor(0, 1);
            LCD_Print("Check connections");
            break;
    }
}

void CarDamage_SendESP32Command(char* command) {
    // Send command to ESP32 via UART
    char cmd_buffer[64];
    sprintf(cmd_buffer, "%s\r\n", command);
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd_buffer, strlen(cmd_buffer), 1000);
}

/**
 * @brief UART receive complete callback
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART1) {
        // Single character received
        if (uart_rx_buffer[uart_rx_index] == '\n' || uart_rx_buffer[uart_rx_index] == '\r') {
            if (uart_rx_index > 0) {
                uart_rx_buffer[uart_rx_index] = '\0'; // Null terminate
                uart_message_ready = 1;
                // Don't increment index for next char, will be reset in processing
            } else {
                // Continue receiving next character at same position
                HAL_UART_Receive_IT(&huart1, &uart_rx_buffer[uart_rx_index], 1);
            }
        } else {
            uart_rx_index++;
            if (uart_rx_index >= sizeof(uart_rx_buffer) - 1) {
                uart_rx_index = 0; // Buffer overflow protection
            }
            // Continue receiving next character
            HAL_UART_Receive_IT(&huart1, &uart_rx_buffer[uart_rx_index], 1);
        }
    }
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
