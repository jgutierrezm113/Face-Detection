
#include "../include/pthreads.h"

using namespace std;
using namespace caffe;
using namespace cv;

void* pnet_thread(void *i) {

  // Process received information regarding queue and scale
  pnet_info info = *((pnet_info *) i);

  #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
  #else
    Caffe::set_mode(Caffe::GPU);
  #endif

  float scale = info.scale;
  Queue<Data*>* queue = info.queue;

  // Create PNET object
  PNet pnet(pnet_model_file, pnet_trained_file);

  while(1){
    // Read packet from queue
    Data* Packet = queue->Remove();

    if (Packet->type == END){
      #if(DEBUG_ENABLED)
        //cout << "PNET stage exiting: " << scale << endl;
      #endif

      break;
    }
    cv::Size input_geometry (ceil(Packet->processed_frame.cols*scale),
                             ceil(Packet->processed_frame.rows*scale));
    pnet.SetInputGeometry(input_geometry);

    // Resize the Image
    std::vector <cv::Mat> img_data;

    cv::Mat resized;

    cv::resize(Packet->processed_frame, resized, input_geometry);

    img_data.push_back(resized);

    // Pnet Input Setup
    pnet.FeedInput(img_data);

    // Pnet Forward data
    pnet.Forward();

    // Release mats
    resized.release();

    std::vector<int> shape;
    std::vector<int>* shape_ptr = &shape;
    std::vector< std::vector <float>> output_data;
    std::vector< std::vector <float>>* output_data_ptr = &output_data;

    pnet.RetrieveOutput(*shape_ptr, *output_data_ptr);

    // Generate Bounding Box based on output from net
    vector<BBox> temp_boxes = generateBoundingBox(output_data,
                                                 shape,
                                                 scale,
                                                 thresholds[0]);
    // Run NMS on boxes
    vector<int> pick = nms (temp_boxes, nms_thresholds[0], 0);

    // Select chosen boxes, update bounding_boxes vector
    vector<BBox> chosen_boxes;
    for (unsigned int j = 0; j < pick.size(); j++){
      chosen_boxes.push_back(temp_boxes[pick[j]]);
    }

    //Update Packet BBoxes
    pthread_mutex_lock(Packet->mut);
    Packet->bounding_boxes.insert(Packet->bounding_boxes.end(), chosen_boxes.begin(), chosen_boxes.end());
    Packet->counter++;
    //if (Packet->pnet_cont == factor_count)
    pthread_cond_signal(Packet->done);
    pthread_mutex_unlock(Packet->mut);
  }

  // Exit
  pthread_exit(0);
}

void* pnet      (void *ptr){

  // Timer
  #if(MEASURE_TIME)
    double start, finish;
  #endif

  // Receive which queue ID its supposed to access
  int queue_id = *((int *) ptr);

  // Receive first frame to know size of image/video

  // Read packet from queue
  Data* Packet = ptr_queue[queue_id].Snoop();

  // Set up PNET stage
  int factor_count = 0;
  float minl = min (Packet->processed_frame.rows, Packet->processed_frame.cols);

  float m = PNET_CONV_SIZE / (float) minSize;

  minl = minl*m;

  // Create Scale Pyramid
  std::vector<float> scales;

  while (minl >= PNET_CONV_SIZE && factor_count < PNET_MAX_SCALE_COUNT){
    scales.push_back(m*pow(factor,factor_count));
    minl *= factor;
    factor_count++;
  }

  // Queue (for sharing resources)
  Queue<Data*>* pnet_queue = new Queue<Data*> [factor_count];

  // Create PNET threads
  pthread_t* pnet_thread_t = new pthread_t [factor_count];
  pnet_info* pnet_info_arg = new pnet_info [factor_count];

  for ( int i = 0; i < factor_count; i++ ) {
    pnet_info_arg[i].scale = scales[i];
    pnet_info_arg[i].queue = &pnet_queue[i];

    pthread_create(&pnet_thread_t[i], 0, pnet_thread, (void *)&pnet_info_arg[i]);
  }

  while (1){

    // Read packet from queue
    Data* Packet = ptr_queue[queue_id].Remove();

    // Record time
    #if(MEASURE_TIME)
      start = CLOCK();
    #endif

    // If Valid == 0; exit pthread
    if (Packet->type == END){
      //cout << "PNET end\n";
      #if(DEBUG_ENABLED)
        printw("Received Valid = 0. Exiting PNET %d stage\n", queue_id);
      #endif

      //Insert packet into PNET queues for exiting (valid = -1)
      for ( int i = 0; i < factor_count; i++ ) {
        pnet_queue[i].Insert(Packet);
      }

      // Wait for child processes to complete
      for ( int i = 0; i < factor_count; i++ ) {
        pthread_join(pnet_thread_t[i], NULL);
      }

      // Send message to next stage
      ptr_queue[queue_id+1].Insert(Packet);
      break;
    }

    // Insert packet into PNET queues for processing
    for ( int i = 0; i < factor_count; i++ ) {
      pnet_queue[i].Insert(Packet);
    }

    // Wait for children to finish processing
    pthread_mutex_lock(Packet->mut);
    while (Packet->counter < factor_count )
      pthread_cond_wait(Packet->done, Packet->mut);
    pthread_mutex_unlock(Packet->mut);

    // Apply NMS again
    if (Packet->bounding_boxes.size() > 0){
      vector<int> pick = nms (Packet->bounding_boxes, nms_thresholds[1], 0);
      // Select chosen boxes, update bounding_boxes vector
      vector<BBox> chosen_boxes;
      for (unsigned int j = 0; j < pick.size(); j++){
        chosen_boxes.push_back(Packet->bounding_boxes[pick[j]]);
      }

      Packet->bounding_boxes.swap(chosen_boxes);

      vector<BBox> correct_box(Packet->bounding_boxes.size());
      for (unsigned int j = 0; j < Packet->bounding_boxes.size(); j++){
        float regw = -Packet->bounding_boxes[j].p2.x+Packet->bounding_boxes[j].p1.x;
        float regh = -Packet->bounding_boxes[j].p2.y+Packet->bounding_boxes[j].p1.y;
        correct_box[j].p1.x = Packet->bounding_boxes[j].p1.x + Packet->bounding_boxes[j].dP1.x*regw;
        correct_box[j].p1.y = Packet->bounding_boxes[j].p1.y + Packet->bounding_boxes[j].dP1.y*regh;
        correct_box[j].p2.x = Packet->bounding_boxes[j].p2.x + Packet->bounding_boxes[j].dP2.x*regw;
        correct_box[j].p2.y = Packet->bounding_boxes[j].p2.y + Packet->bounding_boxes[j].dP2.y*regh;
        correct_box[j].score = Packet->bounding_boxes[j].score;

        // Convert Box to Square (REREQ)
        float h = correct_box[j].p2.y - correct_box[j].p1.y;
        float w = correct_box[j].p2.x - correct_box[j].p1.x;
        float l = max(w, h);

        correct_box[j].p1.x += w*0.5 - l*0.5;
        correct_box[j].p1.y += h*0.5 - l*0.5;
        correct_box[j].p2.x = correct_box[j].p1.x + l;
        correct_box[j].p2.y = correct_box[j].p1.y + l;

        // Fix value to int
        correct_box[j].p1.x = floor(correct_box[j].p1.x);
        correct_box[j].p1.y = floor(correct_box[j].p1.y);
        correct_box[j].p2.x = floor(correct_box[j].p2.x);
        correct_box[j].p2.y = floor(correct_box[j].p2.y);
      }

      Packet->bounding_boxes.swap(correct_box);

      // Pad generated boxes
      padBoundingBox(Packet->bounding_boxes, Packet->processed_frame.rows, Packet->processed_frame.cols);

    }

    // Record time
    #if(MEASURE_TIME)
      finish = CLOCK();
      Packet->stage_time[queue_id] = finish - start;
    #endif

    ptr_queue[queue_id+1].Insert(Packet);
  }

  // Free Resources
  free (pnet_thread_t);
  free (pnet_info_arg);
  // FIXME: Not sure why this free is not working
  //delete [] pnet_queue;

  // Exit
  pthread_exit(0);

}
