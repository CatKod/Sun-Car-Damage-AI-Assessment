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
#include "Timer.h"
#include "lcd1602_i2c.h"
#include "car_damage_comm.h"
#include "lcd_test.h"
#include "i2c_scanner.h"
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
extern timer_Objt Tim_1ms[MaxTIMER];

// Car Damage Detection System Variables
uint32_t last_analysis_request = 0;
uint32_t analysis_interval = 3000; // Request analysis every 3 seconds
uint8_t system_initialized = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

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

  HAL_TIM_Base_Start_IT(&htim2);
  startTim(&Tim_1ms[0], 1000);
  
  // Wait for system to stabilize
  HAL_Delay(1000);
  
  // === I2C SCANNER MODE ===
  // Scan for I2C devices first
  uint8_t devices_found = I2C_ScanAllDevices();
  
  if (devices_found > 0) {
    // Devices found! Signal success and try LCD
    system_initialized = 1;
    
    // Try different common LCD addresses
    uint8_t lcd_addresses[] = {0x27, 0x3F, 0x26, 0x20, 0x38, 0x39};
    uint8_t lcd_found = 0;
    
    for (int i = 0; i < 6; i++) {
      if (I2C_TestAddress(lcd_addresses[i]) == HAL_OK) {
        // Found potential LCD, try to initialize it
        // Temporarily change the address in our library
        // (This is a hack but works for testing)
        lcd_found = 1;
        
        // Try to initialize LCD with this address
        if (LCD_QuickTest() == HAL_OK) {
          // LCD working!
          break;
        }
      }
    }
    
    if (!lcd_found) {
      // I2C devices found but no LCD working
      system_initialized = 2; // Special state
    }
    
  } else {
    // No I2C devices found at all
    system_initialized = 0;
  }
  
  last_analysis_request = HAL_GetTick();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	scanTimer();
	
	// Different LED patterns based on I2C scan results
	static uint32_t led_timer = 0;
	uint32_t current_time = HAL_GetTick();
	
	if (system_initialized == 1) {
		// LCD working - slow steady blink (1Hz)
		if ((current_time - led_timer) >= 500) {
			led_timer = current_time;
			HAL_GPIO_TogglePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin);
		}
		
		// Update LCD every 5 seconds
		if ((current_time - last_analysis_request) >= 5000) {
			last_analysis_request = current_time;
			LCD_Clear();
			LCD_SetCursor(0, 0);
			LCD_Print("  I2C WORKING   ");
			LCD_SetCursor(1, 0);
			LCD_Print("DEVICES: ");
			LCD_PrintInt(num_found);
		}
		
	} else if (system_initialized == 2) {
		// I2C devices found but LCD not working - medium blink (2Hz)
		if ((current_time - led_timer) >= 250) {
			led_timer = current_time;
			HAL_GPIO_TogglePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin);
		}
		
	} else {
		// No I2C devices found - fast blink (5Hz)
		if ((current_time - led_timer) >= 100) {
			led_timer = current_time;
			HAL_GPIO_TogglePin(SYSTEM_LED_GPIO_Port, SYSTEM_LED_Pin);
		}
	}
	
	// Continuous I2C scanning (every 10 seconds)
	static uint32_t scan_timer = 0;
	if ((current_time - scan_timer) >= 10000) {
		scan_timer = current_time;
		
		// Re-scan I2C devices
		uint8_t found = I2C_ScanAllDevices();
		if (found != num_found) {
			// Device count changed, update state
			if (found > 0) {
				system_initialized = (system_initialized == 1) ? 1 : 2;
			} else {
				system_initialized = 0;
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
