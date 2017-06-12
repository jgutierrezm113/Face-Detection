
#include "../include/pthreads.h"

using namespace std;
using namespace caffe;
using namespace cv;
using namespace cv::cuda;

float thresholds[3]     = {0.6, 0.8, 0.95};
//float thresholds[3]     = {0.6, 0.7, 0.7};
float nms_thresholds[3] = {0.5, 0.5, 0.3};
//float nms_thresholds[3] = {0.7, 0.7, 0.7};

// Array of queues (between stages)
Queue <Data*> ptr_queue[STAGE_COUNT];

// PNET
int minSize = 20;
float factor = 0.709;

string pnet_model_file   = "model/det1.prototxt";
string pnet_trained_file = "model/det1.caffemodel";

// RNET
string rnet_model_file   = "model/det2.prototxt";
string rnet_trained_file = "model/det2.caffemodel";

// ONET
string onet_model_file   = "model/det3.prototxt";
string onet_trained_file = "model/det3.caffemodel";

char *file_name;

// Output Variables
bool show_video;
bool record_video;
bool take_snapshot;

// FPS variables for Output Thread
double _avgfps		= 0;
double _avgdur    = 0;

void* preprocess (void *ptr) {

  // Timer
  double start, finish;

  // Receive which queue ID its supposed to access
  int queue_id = *((int *) ptr);

  while (1){
    Data* Packet = ptr_queue[queue_id].Remove();

    if (Packet->type == END){
      if (config.debug) printw("Received Valid = 0. Exiting %d stage\n", queue_id);
      ptr_queue[queue_id+1].Insert(Packet);
      break;
    }

    // Record time
    start = CLOCK();

    // Preprocess Input image (Convert to Float, Normalize, change channels, transpose)
    cv::Mat Matfloat;
    Packet->frame.convertTo(Matfloat, CV_32FC3);

    cv::Mat Normalized;
    cv::normalize(Matfloat, Normalized, -1, 1, cv::NORM_MINMAX, -1);
    cv::cvtColor(Normalized, Normalized, cv::COLOR_BGR2RGB);

    Packet->processed_frame = Normalized.t();

    // Record time
    finish = CLOCK();
    Packet->stage_time[queue_id] = finish - start;

    ptr_queue[queue_id+1].Insert(Packet);

  }

  // Exit
  pthread_exit(0);
}

void* postprocess (void *ptr) {

  // Timer
  double start, finish;

  // Previous Data Packet
  Data Previous;

  // Receive which queue ID its supposed to access
  int queue_id = *((int *) ptr);

  while (1){
    Data* Packet = ptr_queue[queue_id].Remove();

    if (Packet->type == END){
      if (config.debug) printw("Received Valid = 0. Exiting %d stage\n", queue_id);
      ptr_queue[queue_id+1].Insert(Packet);
      break;
    }

    // Record time
    start = CLOCK();

    // Correct box coordinates to the original image
    for (unsigned int j = 0; j < Packet->bounding_boxes.size(); j++){
      cv::Point2f temp;
      temp = Packet->bounding_boxes[j].p1;
      Packet->bounding_boxes[j].p1.x = temp.y;
      Packet->bounding_boxes[j].p1.y = temp.x;

      temp = Packet->bounding_boxes[j].p2;
      Packet->bounding_boxes[j].p2.x = temp.y;
      Packet->bounding_boxes[j].p2.y = temp.x;
    }

    // Correct landmark coordinates to the original image
    for (unsigned int j = 0; j < Packet->landmarks.size(); j++){

      // Create Points for box
      Landmark points;

      points = Packet->landmarks[j];

      Packet->landmarks[j].LE.x = points.LE.y;
      Packet->landmarks[j].RE.x = points.RE.y;
      Packet->landmarks[j].N.x  = points.N.y;
      Packet->landmarks[j].LM.x = points.LM.y;
      Packet->landmarks[j].RM.x = points.RM.y;

      Packet->landmarks[j].LE.y = points.LE.x;
      Packet->landmarks[j].RE.y = points.RE.x;
      Packet->landmarks[j].N.y  = points.N.x;
      Packet->landmarks[j].LM.y = points.LM.x;
      Packet->landmarks[j].RM.y = points.RM.x;
    }

    // Get feedback from previous frame
    for (unsigned int j = 0; j < Packet->bounding_boxes.size(); j++){
      for (unsigned int k = 0; k < Previous.bounding_boxes.size(); k++){
        float Narea = abs(Packet->bounding_boxes[j].p2.x - Packet->bounding_boxes[j].p1.x)*
                      abs(Packet->bounding_boxes[j].p2.y - Packet->bounding_boxes[j].p1.y);
        float Parea = abs(Previous.bounding_boxes[k].p2.x - Previous.bounding_boxes[k].p1.x)*
                      abs(Previous.bounding_boxes[k].p2.y - Previous.bounding_boxes[k].p1.y);
        float   xx1 = max(Packet->bounding_boxes[j].p1.x,Previous.bounding_boxes[k].p1.x);
        float   yy1 = max(Packet->bounding_boxes[j].p1.y,Previous.bounding_boxes[k].p1.y);
        float   xx2 = min(Packet->bounding_boxes[j].p2.x,Previous.bounding_boxes[k].p2.x);
        float   yy2 = min(Packet->bounding_boxes[j].p2.y,Previous.bounding_boxes[k].p2.y);
        float     w = max( 0.0f, (xx2-xx1));
        float     h = max( 0.0f, (yy2-yy1));
        float inter = w * h;
        float   iou = inter/(Narea + Parea - inter);

        // If Over threshold, similar enough. average them
        if (iou > 0.3){
          Packet->bounding_boxes[j].p1.x = (Packet->bounding_boxes[j].p1.x + Previous.bounding_boxes[k].p1.x)/2;
          Packet->bounding_boxes[j].p1.y = (Packet->bounding_boxes[j].p1.y + Previous.bounding_boxes[k].p1.y)/2;
          Packet->bounding_boxes[j].p2.x = (Packet->bounding_boxes[j].p2.x + Previous.bounding_boxes[k].p2.x)/2;
          Packet->bounding_boxes[j].p2.y = (Packet->bounding_boxes[j].p2.y + Previous.bounding_boxes[k].p2.y)/2;

          // Not averaging landmarks. Doesn't work well
        }
      }
    }

    // Save previous frame
    Previous = *Packet;

    // Record time
    finish = CLOCK();
    Packet->stage_time[queue_id] = finish - start;

    ptr_queue[queue_id+1].Insert(Packet);

  }

  // Exit
  pthread_exit(0);
}

void* output (void *ptr) {

  // Timer
  double start, finish;

  // OPENCV window thread for closing window
  cv::startWindowThread();

  // Receive which queue ID its supposed to access
  int queue_id = *((int *) ptr);

  // Will be used for img window and file writing
  string name = config.short_file_name;

  // Local variables that serve as memories in case of swift changes
  bool local_show_video    = 0;
  bool local_log_results   = 0;
  bool local_record_video  = 0;
  long int local_log_frames = 0;

  // Output streams for files
  cv::VideoWriter outputVideo;
  std::ofstream ofs;

  if (config.show_video){
    namedWindow(name.c_str(),CV_WINDOW_NORMAL); //create a window
    local_show_video = 1;
  }

  // Initialize AVG counters
  avginit();

  while (1){
    Data* Packet = ptr_queue[queue_id].Remove();

    if (Packet->type == END){
      delete Packet;
      if (config.debug) printw("Received Valid = 0. Exiting %d stage\n", queue_id);
      break;
    }

    // Increase frame counter
    local_log_frames++;

    // Record time
    start = CLOCK();

    // Add boxes and features to frame
    writeOutputImage(Packet);

    // Control Display
    if (!config.show_video && local_show_video){
      destroyAllWindows();
      //waitKey(1);
      local_show_video = 0;
    } else if (config.show_video && !local_show_video){
      namedWindow(name.c_str(),CV_WINDOW_NORMAL); //create a window
      local_show_video = 1;
    }

    // Show display
    if (local_show_video){
      // Open window with detected objects
      imshow(name.c_str(), Packet->frame);
      resizeWindow(name.c_str(), 640, 480);

      // FIXME: BUG in GTK wont allow to use waitKey
      // Image wont be desplayed when its a single image
      //waitKey(1);
      /* if (Packet->type == IMG){
              //cv::waitKey();
      } */
    }

    // Save snapshot
    if (config.take_snapshot){
      config.take_snapshot = 0;
      // Write timestamp on name
      time_t rawtime;
      struct tm * timeinfo;
      char buffer[80];

      time (&rawtime);
      timeinfo = localtime(&rawtime);

      strftime(buffer,sizeof(buffer),"_%Y_%m_%d_-_%I_%M_%S",timeinfo);
      std::string timestamp(buffer);

      // Write output
      stringstream ss;
      ss << "outputs/" << config.short_file_name << timestamp << ".jpg";
      string commS = ss.str();
      const char* comm = commS.c_str();
      cout << "Writing " << comm << endl;
      cv::imwrite(comm, Packet->frame);
    }

    // Save video
    if (config.type == CAM || config.type == VID){
      if (config.record_video && !local_record_video){
        local_record_video = 1;

        // Set file name using timestamp
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer,sizeof(buffer),"_%Y_%m_%d_-_%I_%M_%S",timeinfo);
        std::string timestamp(buffer);

        stringstream ss;
        ss << "outputs/" << config.short_file_name << timestamp << ".avi";
        string commS = ss.str();
        const char* comm = commS.c_str();

        Size S = Size((int) Packet->frame.cols,    // Acquire input size
        (int) Packet->frame.rows);

        outputVideo.open(comm, CV_FOURCC('M','J','P','G'), fps, S, true);

        if (!outputVideo.isOpened()){
          printw("Could not open the output video for write: %s \n", comm);
          local_record_video = 0;
          config.record_video = 0;
        }
        outputVideo << Packet->frame;

      } else if (!config.record_video && local_record_video){
        local_record_video = 0;
        outputVideo.release();
      } else if (config.record_video && local_record_video){
        outputVideo << Packet->frame;
      }
    }

    // Record time
   	Packet->end_time = CLOCK();
    finish = CLOCK();
    Packet->stage_time[queue_id] = finish - start;

    // Print metrics
    std::stringstream oss;
    _avgfps = avgfps(_avgfps);
    double total_time = Packet->end_time - Packet->start_time;
    _avgdur = avgdur(total_time, _avgdur);

    oss << "--DATA--" << endl;
    if (config.type != IMG){
      oss << "Average FPS:  " << _avgfps << endl
          << "Average Time: " << _avgdur << endl;
    }
    oss << "Total Time:   " << total_time << endl
        << "PreP Time:    " << Packet->stage_time[0] << endl
        << "PNET Time:    " << Packet->stage_time[1] << endl
        << "RNET Time:    " << Packet->stage_time[2] << endl
        << "ONET Time:    " << Packet->stage_time[3] << endl
        << "PostP Time:   " << Packet->stage_time[4] << endl
        << "Output Time:  " << Packet->stage_time[5] << endl
        << endl;
    output_string = oss.str();

    // Write metrics to file
    if (config.log_results && !local_log_results){
      local_log_results = 1;

      // Set file name using timestamp
      time_t rawtime;
      struct tm * timeinfo;
      char buffer[80];

      time (&rawtime);
      timeinfo = localtime(&rawtime);

      strftime(buffer,sizeof(buffer),"_%Y_%m_%d_-_%I_%M_%S",timeinfo);
      std::string timestamp(buffer);

      stringstream ss;
      ss << "outputs/" << config.short_file_name << timestamp << ".csv";
      string commS = ss.str();
      const char* comm = commS.c_str();

      // Open File for write
      ofs.open (comm, std::ofstream::out | std::ofstream::app);

      if (!ofs.is_open()){
        cout << "Unable to open file " << comm << " for writing." << endl;
        config.log_results = 0;
        local_log_results = 0;
      } else {
        cout << "Writing " << comm << endl;

        // Write First Row (with metrics)
        ofs << "Frame Number,";
        if (config.type != IMG){
          ofs << "Average FPS,Average Time,";
        }
        ofs << "Total Time,PreP Time,PNET Time,RNET Time,ONET Time,PostP Time,Output Time" << endl;

        // Write First Array
        ofs << local_log_frames << ",";
        if (config.type != IMG){
          ofs << _avgfps << "," << _avgdur << ",";
        }
        ofs << total_time << ","
        << Packet->stage_time[0] << ","
        << Packet->stage_time[1] << ","
        << Packet->stage_time[2] << ","
        << Packet->stage_time[3] << ","
        << Packet->stage_time[4] << ","
        << Packet->stage_time[5] << ","
        << endl;
      }
    } else if (!config.log_results && local_log_results){
      local_log_results = 0;
      ofs.close();
    } else if (config.log_results && local_log_results){
      ofs << local_log_frames << ","
          << _avgfps << ","
          << _avgdur << ","
          << total_time << ","
          << Packet->stage_time[0] << ","
          << Packet->stage_time[1] << ","
          << Packet->stage_time[2] << ","
          << Packet->stage_time[3] << ","
          << Packet->stage_time[4] << ","
          << Packet->stage_time[5] << ","
          << endl;
    }

    if (config.type == IMG){
      cout << output_string.c_str();
    }

    delete Packet;
  }

  // Exit
  destroyAllWindows();
  pthread_exit(0);
}
