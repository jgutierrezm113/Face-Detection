
#include "../include/pthreads.h"

using namespace std;
using namespace caffe;
using namespace cv;

void* onet (void *ptr){

	// Timer
	double start, finish;

	// Receive which queue ID its supposed to access
	int queue_id = *((int *) ptr);

	#ifdef CPU_ONLY
		Caffe::set_mode(Caffe::CPU);
	#else
		Caffe::set_mode(Caffe::GPU);
	#endif

	// Create RNET object
	ONet onet(onet_model_file, onet_trained_file);

	cv::Size input_geometry(48, 48);

	onet.SetInputGeometry(input_geometry);

	// Read StartupPacket
	Data* Packet = ptr_queue[queue_id].Remove();

	//Update Packet to say this thread is ready
	if (Packet->type == STU)
		Packet->IncreaseCounter();

	while (1){

		// Read packet from queue
		Data* Packet = ptr_queue[queue_id].Remove();

		if (Packet->type == END || Packet->type == ILL){
			if (config.debug) printw("Received Valid = 0. Exiting %d stage\n", queue_id);

			// Send message to next stage
			ptr_queue[queue_id+1].Insert(Packet);
			break;
		}

		// Record time
		start = CLOCK();

		if (Packet->bounding_boxes.size() > 0){

			// Vector of cropped images
			vector<cv::Mat> cropBoxes;

			// Generate cropped images from the main image
			for (unsigned int i = 0; i < Packet->bounding_boxes.size(); i++) {

				cv::Rect rect =  cv::Rect(Packet->bounding_boxes[i].p1.x,
																	Packet->bounding_boxes[i].p1.y,
																	Packet->bounding_boxes[i].p2.x - Packet->bounding_boxes[i].p1.x, //width
																	Packet->bounding_boxes[i].p2.y - Packet->bounding_boxes[i].p1.y) //height
											 & cv::Rect(0, 0, Packet->processed_frame.cols, Packet->processed_frame.rows);

				cv::Mat crop = cv::Mat(Packet->processed_frame, rect).clone();

				// Resize the cropped Image
				cv::Mat img_data;

				cv::resize(crop, img_data, input_geometry);

				cropBoxes.push_back(img_data);

				img_data.release();
			}

			// Onet Input Setup
			onet.FeedInput(cropBoxes);

			// Onet Forward data
			onet.Forward();

			std::vector<int> shape;
			std::vector<int>* shape_ptr = &shape;
			std::vector< std::vector <float>> output_data;
			std::vector< std::vector <float>>* output_data_ptr = &output_data;

			onet.RetrieveOutput(*shape_ptr, *output_data_ptr);

			// Filter Boxes that are over threshold and collect mv output values as well
			vector<BBox> chosen_boxes;
			for (int j = 0; j < shape[0]; j++){ // same as num boxes
				if (output_data[2][j*2+1] > thresholds[2]){

					// Saving mv output data in boxes extra information
					Packet->bounding_boxes[j].dP1.x = output_data[0][j*4+1];
					Packet->bounding_boxes[j].dP1.y = output_data[0][j*4+0];
					Packet->bounding_boxes[j].dP2.x = output_data[0][j*4+3];
					Packet->bounding_boxes[j].dP2.y = output_data[0][j*4+2];
					Packet->bounding_boxes[j].score = output_data[2][j*2+1];
					chosen_boxes.push_back(Packet->bounding_boxes[j]);

					// Create Points for box
					Landmark points;

					float w = Packet->bounding_boxes[j].p2.x - Packet->bounding_boxes[j].p1.x;
					float h = Packet->bounding_boxes[j].p2.y - Packet->bounding_boxes[j].p1.y;

					points.LE.x = w*output_data[1][j*10+5] + Packet->bounding_boxes[j].p1.x;
					points.RE.x = w*output_data[1][j*10+6] + Packet->bounding_boxes[j].p1.x;
					points.N.x  = w*output_data[1][j*10+7] + Packet->bounding_boxes[j].p1.x;
					points.LM.x = w*output_data[1][j*10+8] + Packet->bounding_boxes[j].p1.x;
					points.RM.x = w*output_data[1][j*10+9] + Packet->bounding_boxes[j].p1.x;

					points.LE.y = h*output_data[1][j*10+0] + Packet->bounding_boxes[j].p1.y;
					points.RE.y = h*output_data[1][j*10+1] + Packet->bounding_boxes[j].p1.y;
					points.N.y  = h*output_data[1][j*10+2] + Packet->bounding_boxes[j].p1.y;
					points.LM.y = h*output_data[1][j*10+3] + Packet->bounding_boxes[j].p1.y;
					points.RM.y = h*output_data[1][j*10+4] + Packet->bounding_boxes[j].p1.y;

					Packet->landmarks.push_back(points);
				}
			}
			Packet->bounding_boxes.swap(chosen_boxes);

		}
		if (Packet->bounding_boxes.size() > 0){

			vector<BBox> correct_box(Packet->bounding_boxes.size());
			for (unsigned int j = 0; j < Packet->bounding_boxes.size(); j++){

				// Apply BBREG
				float regw = Packet->bounding_boxes[j].p2.x-Packet->bounding_boxes[j].p1.x;
				float regh = Packet->bounding_boxes[j].p2.y-Packet->bounding_boxes[j].p1.y;

				correct_box[j].p1.x = Packet->bounding_boxes[j].p1.x + bbox_adjust_percentage*Packet->bounding_boxes[j].dP1.x*regw;
				correct_box[j].p1.y = Packet->bounding_boxes[j].p1.y + bbox_adjust_percentage*Packet->bounding_boxes[j].dP1.y*regh;
				correct_box[j].p2.x = Packet->bounding_boxes[j].p2.x + bbox_adjust_percentage*Packet->bounding_boxes[j].dP2.x*regw;
				correct_box[j].p2.y = Packet->bounding_boxes[j].p2.y + bbox_adjust_percentage*Packet->bounding_boxes[j].dP2.y*regh;
				correct_box[j].score = Packet->bounding_boxes[j].score;
			}
			Packet->bounding_boxes.swap(correct_box);

			vector<int> pick = nms (Packet->bounding_boxes, nms_thresholds[2], 1);
			// Select chosen boxes, update bounding_boxes vector
			vector<BBox> chosen_boxes;
			vector<Landmark> chosen_points;
			for (unsigned int j = 0; j < pick.size(); j++){
				chosen_boxes.push_back(Packet->bounding_boxes[pick[j]]);
				chosen_points.push_back(Packet->landmarks[pick[j]]);
			}

			Packet->bounding_boxes.swap(chosen_boxes);
			Packet->landmarks.swap(chosen_points);

		}

		// Record time
		finish = CLOCK();
		Packet->stage_time[queue_id] = finish - start;

		ptr_queue[queue_id+1].Insert(Packet);
	}

	// Exit
	pthread_exit(0);

}
