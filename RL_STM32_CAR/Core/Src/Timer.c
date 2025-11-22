#include "Timer.h"
#include "tim.h"
#include "stdlib.h"

timer_Objt Tim_1ms[MaxTIMER];
uint32_t tim1msTick;
void startTim(timer_Objt *pTim, uint32_t SV)
{
	if(pTim->En == 0)
	{
		pTim->En = 1;
		if(SV<1) SV = 1;
		pTim->SV = SV-1;
		pTim->PV = 0;
		pTim->Output = 0;
	}
}

void stopTim(timer_Objt *pTim)
{
	pTim->En = 0;
	pTim->SV = 0;
	pTim->PV = 0;
	pTim->Output = 0;
}

/**********************************************************************************************/
//void scanTime1ms(void)
void scanTimer(void)
{
	if(tim1msTick == 1){
		tim1msTick = 0;
		int i = 0;
		for(i=0; i<MaxTIMER; i++)									//Scan all declared timers
		{
			if(Tim_1ms[i].En == 1)
			{
				if(++Tim_1ms[i].PV > Tim_1ms[i].SV)	//Overflow
				{
					Tim_1ms[i].PV = 0;
					Tim_1ms[i].Output = 1;						//Turn On the Timer Output
				}
			}
		}
	}
}

/**********************************************************************************************/
void Timer_IT_Init(void)
{
	HAL_TIM_Base_Start_IT(&htim2); //timer 1ms
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if(htim->Instance == TIM2)		//Timer 2 interrupt every 10ms
	{
		tim1msTick = 1; 
	}
}