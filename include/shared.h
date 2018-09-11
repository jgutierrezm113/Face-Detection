#ifndef SHARED_H
#define SHARED_H

#include "def.h"

// Declaration of shared global variables
extern Queue <Data*> ptr_queue[STAGE_COUNT];
extern float thresholds[3];
extern float nms_thresholds[3];
extern int minSize;
extern float factor;
extern float bbox_adjust_percentage;
extern string pnet_model_file;
extern string rnet_model_file;
extern string onet_model_file;

extern string pnet_trained_file;
extern string rnet_trained_file;
extern string onet_trained_file;

extern double fps;

extern string output_string;

#ifdef SEQUENTIAL_ON
	extern Data seq_contr;
#endif

#endif
