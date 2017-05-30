
#include "../include/pthreads.h"

using namespace std;
using namespace caffe;
using namespace cv;

float thresholds[3]     = {0.6, 0.6, 0.95};
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

void* preprocess (void *ptr) {
        
        // Receive which queue ID its supposed to access
        int queue_id = *((int *) ptr);
        
        while (1){
                Data* Packet = ptr_queue[queue_id].Remove();

                if (Packet->type == END){
                        #if(DEBUG_ENABLED)
                                printw("Received Valid = 0. Exiting %d stage\n", queue_id);
                        #endif
                        ptr_queue[queue_id+1].Insert(Packet);
                        break;                        
                }
                
                // Preprocess Input image (Convert to Float, Normalize, change channels, transpose)
                cv::Mat Matfloat;
                Packet->frame.convertTo(Matfloat, CV_32FC3);
                
                cv::Mat Normalized;
                cv::normalize(Matfloat, Normalized, -1, 1, cv::NORM_MINMAX, -1);
                cv::cvtColor(Normalized, Normalized, cv::COLOR_BGR2RGB);
                
                Packet->processed_frame = Normalized.t();
                
                ptr_queue[queue_id+1].Insert(Packet);

        }
        
        // Exit
        pthread_exit(0);
}

void* output (void *ptr) {
          
        // OPENCV window thread for closing window
        cv::startWindowThread();
           
        // Receive which queue ID its supposed to access
        int queue_id = *((int *) ptr);
        
        // Will be used for img window and file writing
        string name = config.short_file_name;
        
        // Local variables that serve as memories in case of swift changes
        bool local_show_video    = 0;
        bool local_record_video  = 0;
        cv::VideoWriter outputVideo;
        
        if (config.show_video){
                namedWindow(name.c_str(),CV_WINDOW_NORMAL); //create a window
                local_show_video = 1;
        }
        while (1){
                Data* Packet = ptr_queue[queue_id].Remove();
                
                if (Packet->type == END){
                        delete Packet;
                        #if(DEBUG_ENABLED)
                                printw("Received Valid = 0. Exiting %d stage\n", queue_id);
                        #endif
                        break;                        
                }

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
                        cout << "writing " << comm << endl;
                        cv::imwrite(comm, Packet->frame);
                }                
                
                // Save video
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
                
                delete Packet;
                
        }
        
        // Exit
        destroyAllWindows();
        pthread_exit(0);
}
