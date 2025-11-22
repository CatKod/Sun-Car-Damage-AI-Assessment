#ifndef		__tim_h
#define		__tim_h

#include "stdint.h"

#define MaxTIMER 20

typedef struct
{
	uint32_t En;
	uint32_t SV;
	uint32_t PV;
	uint32_t Output;
}timer_Objt;

void startTim(timer_Objt *pTim, uint32_t SV);	//Start timer 10ms
void stopTim(timer_Objt *pTim);								//Reset timer 10ms
void scanTimer(void);
void Timer_IT_Init(void);
#endif