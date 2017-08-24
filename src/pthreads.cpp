
#include "../include/pthreads.h"

using namespace std;
using namespace caffe;
using namespace cv;
//using namespace cv::cuda;

float thresholds[3]     = {0.6, 0.8, 0.85};
float nms_thresholds[3] = {0.4, 0.4, 0.3};

// For FDDB
// float thresholds[3]     = {0.5, 0.8, 0.85};
// float nms_thresholds[3] = {0.5, 0.5, 0.3};

// Original
// float thresholds[3]     = {0.5, 0.5, 0.3};
// float nms_thresholds[3] = {0.5, 0.7, 0.7};

// Array of queues (between stages)
Queue <Data*> ptr_queue[STAGE_COUNT];

// PNET
// int minSize = 10;
// float factor = 0.79;

int minSize = 20;
float factor = 0.709;

// ONET
// Bounding Box Regression effect (to reduce it)
float bbox_adjust_percentage = 1;

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

  // Read StartupPacket
  Data* Packet = ptr_queue[queue_id].Remove();

  //Update Packet to say this thread is ready
  if (Packet->type == STU)
    Packet->IncreaseCounter();

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

    if (Normalized.channels() == 3 || Normalized.channels() == 4 )
      cv::cvtColor(Normalized, Normalized, cv::COLOR_BGR2RGB);
    else if (Normalized.channels() == 1)
      cv::cvtColor(Normalized, Normalized, cv::COLOR_GRAY2RGB);

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

  // Read StartupPacket
  Data* Packet = ptr_queue[queue_id].Remove();

  //Update Packet to say this thread is ready
  if (Packet->type == STU)
    Packet->IncreaseCounter();

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
  double total_time;

  // Receive which queue ID its supposed to access
  int queue_id = *((int *) ptr);

  // File name used
  std::string file_name;

  // Will be used for img window and file writing
  if (config.type == VID){
    file_name = config.full_file_name;
    const std::regex slashrm(".*/");
    std::stringstream result;
    std::regex_replace(std::ostream_iterator<char>(result), file_name.begin(), file_name.end(), slashrm, "");
    file_name = result.str();
    file_name = file_name.substr(0, file_name.find_last_of("."));

  } else if (config.type != VID || config.type != CAM ){
    file_name = config.full_file_name;
    const std::regex slashrm(".*/");
    std::stringstream result;
    std::regex_replace(std::ostream_iterator<char>(result), file_name.begin(), file_name.end(), slashrm, "");
    file_name = result.str();
  } else {
    file_name = "CAM";
  }

  if (config.type != DTB && config.type != IMG){
    // OPENCV window thread for closing window
    cv::startWindowThread();
  }

  // Local variables that serve as memories in case of swift changes
  bool local_show_video     = 0;
  bool local_log_results    = 0;
  bool local_fddb_results   = 0;
  bool local_record_video   = 0;
  long int local_log_frames = 0;

  // Output streams for files
  cv::VideoWriter outputVideo;
  std::ofstream fddb_ofs;
  std::ofstream log_ofs;

  if (config.show_video){
    namedWindow(file_name.c_str(),CV_WINDOW_NORMAL); //create a window
    local_show_video = 1;
  }

  // Initialize AVG counters
  avginit();

  // Read StartupPacket
  Data* Packet = ptr_queue[queue_id].Remove();

  //Update Packet to say this thread is ready
  if (Packet->type == STU)
    Packet->IncreaseCounter();

  while (1){

    Data* Packet = ptr_queue[queue_id].Remove();

    if (Packet->type == END){
      delete Packet;
      if (config.debug) printw("Received Valid = 0. Exiting %d stage\n", queue_id);
      break;
    }

    // Record time
    start = CLOCK();

    // Increase frame counter
    local_log_frames++;

    // Control Display
    if (!config.show_video && local_show_video){
      destroyAllWindows();
      local_show_video = 0;

    } else if (config.show_video && !local_show_video){
      namedWindow(file_name.c_str(),CV_WINDOW_NORMAL); //create a window
      local_show_video = 1;

    }

    // Add BBoxes to image
    writeOutputImage(Packet);

    // Show Display
    if (local_show_video){

      // Open window with detected objects
      imshow(file_name.c_str(), Packet->frame);
      resizeWindow(file_name.c_str(), 640, 480);

      // FIXME: BUG in GTK wont allow to use waitKey
      // Image wont be desplayed when its a single image
      /* if (Packet->type == IMG){
              //cv::waitKey();
      } */
    }

    // Save snapshot
    if (config.take_snapshot){
      if(config.type != DTB)
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
      std::string text;
      if (config.type == DTB)
        text = Packet->name;
      else // IMG use file_name
        text = file_name;
      const std::regex slashrm(".*/");
      std::stringstream result;
      std::regex_replace(std::ostream_iterator<char>(result), text.begin(), text.end(), slashrm, "");
      text = result.str();
      text = text.substr(0, text.find_last_of("."));

      stringstream ss;
      ss << config.output_dir << "/" << text << timestamp << ".jpg";
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
        ss << config.output_dir << "/" << file_name << timestamp << ".avi";
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

    // Recollect Metrics
    if(config.verbose || config.debug){
      std::stringstream oss;
      _avgfps = avgfps(_avgfps);
      total_time = Packet->end_time - Packet->start_time;
      _avgdur = avgdur(total_time, _avgdur);

      oss << std::setprecision(3) << std::fixed << "Data\n----" << endl;
      if (config.type != IMG){
        oss << "Average FPS :  " << _avgfps << endl
          <<   "Average Time:  " << _avgdur << endl;
      }
      oss << "Total Time  :  " << total_time << endl
          << "Main Time   :  " << Packet->stage_time[6] << endl
          << "PreP Time   :  " << Packet->stage_time[0] << endl
          << "PNET Time   :  " << Packet->stage_time[1] << endl
          << "RNET Time   :  " << Packet->stage_time[2] << endl
          << "ONET Time   :  " << Packet->stage_time[3] << endl
          << "PostP Time  :  " << Packet->stage_time[4] << endl
          << "Output Time :  " << Packet->stage_time[5] << endl
          << endl;
      output_string = oss.str();

      // Print information on the processed image
      if (config.type == IMG)
        cout << output_string.c_str();
    }

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
      ss << config.output_dir << "/" << file_name << timestamp << ".csv";
      string commS = ss.str();
      const char* comm = commS.c_str();

      // Open File for write
      log_ofs.open (comm, std::ofstream::out | std::ofstream::app);

      if (!log_ofs.is_open()){
        cout << "Unable to open file " << comm << " for writing." << endl;
        config.log_results = 0;
        local_log_results = 0;
      } else {
        cout << "Writing " << comm << endl;

        // Write First Row (with metrics)
        if (config.type == VID)
          log_ofs << "Frame Number,";
        else
          log_ofs << "Image name,";

        if (config.type != IMG)
          log_ofs << "Average FPS,Average Time,";

        log_ofs << "Total Time,Main Time,PreP Time,PNET Time,RNET Time,ONET Time,PostP Time,Output Time" << endl;

        // Write First Array
        if (config.type == VID)
          log_ofs << local_log_frames << ",";
        else
          log_ofs << Packet->name << ",";

        if (config.type != IMG){
          log_ofs << _avgfps << "," << _avgdur << ",";
        }
        log_ofs << total_time << ","
        << Packet->stage_time[6] << ","
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
      log_ofs.close();
    } else if (config.log_results && local_log_results){
      if (config.type == VID)
        log_ofs << local_log_frames << ",";
      else
        log_ofs << Packet->name << ",";

      log_ofs << _avgfps << ","
          << _avgdur << ","
          << total_time << ","
          << Packet->stage_time[6] << ","
          << Packet->stage_time[0] << ","
          << Packet->stage_time[1] << ","
          << Packet->stage_time[2] << ","
          << Packet->stage_time[3] << ","
          << Packet->stage_time[4] << ","
          << Packet->stage_time[5] << ","
          << endl;
    }

    // Write fddb to file
    if (config.fddb_results && !local_fddb_results){
      local_fddb_results = 1;

      // Set file name using timestamp
      time_t rawtime;
      struct tm * timeinfo;
      char buffer[80];

      time (&rawtime);
      timeinfo = localtime(&rawtime);

      strftime(buffer,sizeof(buffer),"_%Y_%m_%d_-_%I_%M_%S",timeinfo);
      std::string timestamp(buffer);

      stringstream ss;
      ss << config.output_dir << "/" << file_name << timestamp << ".txt";
      string commS = ss.str();
      const char* comm = commS.c_str();

      // Open File for write
      fddb_ofs.open (comm, std::ofstream::out | std::ofstream::app);

      if (!fddb_ofs.is_open()){
        cout << "Unable to open file " << comm << " for writing." << endl;
        config.fddb_results = 0;
        local_fddb_results = 0;
      } else {
        cout << "Writing " << comm << endl;

        // Write Output
        fddb_ofs << Packet->name << endl;
        fddb_ofs << Packet->bounding_boxes.size() << endl;
        for (uint i = 0; i< Packet->bounding_boxes.size(); i++){
          float width  = abs(Packet->bounding_boxes[i].p1.x -
                             Packet->bounding_boxes[i].p2.x);
          float height = abs(Packet->bounding_boxes[i].p1.y -
                             Packet->bounding_boxes[i].p2.y);
          float left   = min(Packet->bounding_boxes[i].p1.x,
                             Packet->bounding_boxes[i].p2.x);
          float top    = min(Packet->bounding_boxes[i].p1.y,
                             Packet->bounding_boxes[i].p2.y);
          float score  = Packet->bounding_boxes[i].score;
          fddb_ofs << left << " " << top << " " << width << " "
              << height << " " << score << endl;
        }
      }
    } else if (!config.fddb_results && local_fddb_results){
      local_fddb_results = 0;
      fddb_ofs.close();
    } else if (config.fddb_results && local_fddb_results){
      // Write Output
      fddb_ofs << Packet->name << endl;
      fddb_ofs << Packet->bounding_boxes.size() << endl;
      for (uint i = 0; i< Packet->bounding_boxes.size(); i++){
        float width  = abs(Packet->bounding_boxes[i].p1.x -
                           Packet->bounding_boxes[i].p2.x);
        float height = abs(Packet->bounding_boxes[i].p1.y -
                           Packet->bounding_boxes[i].p2.y);
        float left   = min(Packet->bounding_boxes[i].p1.x,
                           Packet->bounding_boxes[i].p2.x);
        float top    = min(Packet->bounding_boxes[i].p1.y,
                           Packet->bounding_boxes[i].p2.y);
        float score  = Packet->bounding_boxes[i].score;
        fddb_ofs << left << " " << top << " " << width << " "
            << height << " " << score << endl;
      }
    }

    delete Packet;
  }

  // Exit
  destroyAllWindows();
  pthread_exit(0);
}
