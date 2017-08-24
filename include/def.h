#ifndef DEF_H
#define DEF_H

#include "include.h"
#include "queue.h"

//#define CPU_ONLY

//#define SEQUENTIAL_ON

// Size of the convolution box (Don't change)
#define PNET_CONV_SIZE 12

// Limit max amount of threads running PNET stage
#define PNET_MAX_SCALE_COUNT 12

// CAM properties
#define CAM_FPS 25
#define FRAME_WIDTH 1920
#define FRAME_HEIGHT 1080

// Number of stages in the pipeline
#define STAGE_COUNT 6

enum package_type { STU,  // Start Up Package
										IMG,  // Image Package
										VID,  // Video Package
										CAM,  // Camera Package
										DTB,  // Database Package
										END,  // Ending Package
										ILL}; // Illegal Package

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

typedef struct conf {
	package_type type;
	bool verbose;
	bool debug;
	bool show_video;
	bool record_video;

	// To store results in log file
	bool log_results;
	bool fddb_results;

	// CAM ID
	int cam_id;

	// Only when running
	bool take_snapshot;

	// File name for output writes
	char *full_file_name;
	std::string short_file_name;

	// Database related information
	char *image_dir;

	// output directory
	char *output_dir;

} CONF;

extern CONF config;

#endif
