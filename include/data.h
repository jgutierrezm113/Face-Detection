#ifndef DATA_H
#define DATA_H

#include "include.h"
#include "def.h"

class Data {
	public:
		// File name
		std::string name;

		// Type indicator
		package_type type;

		// Actual Data
		cv::Mat frame;
		cv::Mat processed_frame;

		std::vector <BBox> bounding_boxes;
		std::vector <Landmark> landmarks;

		// Synchronization Variables
		int counter;
		pthread_mutex_t mut;
		pthread_cond_t done;

		// Timing variables
		double start_time;
		double end_time;
		double stage_time[STAGE_COUNT+1]; // PreP-PNET-RNET-ONET-PostP-Out-Main

		// Functions
		void WaitForCounter(int num);
		void IncreaseCounter(void);
		void ResetCounter(void);

		Data();
		~Data();
};

#endif
