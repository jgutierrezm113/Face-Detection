#ifndef DEF_H
#define DEF_H

#include "include.h"

//#define CPU_ONLY

// Maximum Elements in the Queue
#define QUEUESIZE 10

// Size of the convolution box (Don't change)
#define PNET_CONV_SIZE 12

// Number of stages in the pipeline
#define STAGE_COUNT 5 

#define DEBUG_ENABLED (0)

enum Processing_Type { IMG, VID, CAM, DTB, TRAIN, END};

typedef struct {
       // Bounding Box
       cv::Point2f p1;
       cv::Point2f p2;
       
       // Score
       float score;
       
       // Bounding Box Regression adjustment
       cv::Point2f dP1;
       cv::Point2f dP2;
} BBox;

typedef struct {
       cv::Point2f LE;
       cv::Point2f RE;
       cv::Point2f N;
       cv::Point2f LM;
       cv::Point2f RM;
} Landmark;

#endif
