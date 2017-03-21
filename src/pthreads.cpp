
#include "../include/pthreads.h"

using namespace std;
using namespace caffe;
using namespace cv;

// timing variables in main
extern clock_t cbegin;
extern clock_t cend;

float thresholds[3]     = {0.6, 0.6, 0.8};
//float thresholds[3]     = {0, 0, 0};
float nms_thresholds[3] = {0.8, 0.7, 0.3};
//float nms_thresholds[3] = {0, 0, 0};

// Array of queues (between stages)
Queue ptr_queue[STAGE_COUNT];

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

void* preprocess (void *ptr) {
        
        // Receive which queue ID its supposed to access
        int queue_id = *((int *) ptr);
        
        while (1){
                Data* Packet = ptr_queue[queue_id].Remove();

                if (Packet->type == END){
                        #if(DEBUG_ENABLED)
                                cout << "Received Valid = 0. Exiting " << queue_id << " stage\n";
                        #endif
                        ptr_queue[queue_id+1].Insert(Packet);
                        break;                        
                }
                
                // Apply Histogram Equalization
                if(Packet->frame.channels() >= 3) {
                        Mat ycrcb;

                        cvtColor(Packet->frame,ycrcb,CV_BGR2YCrCb);

                        vector<Mat> channels;
                        split(ycrcb,channels);

                        equalizeHist(channels[0], channels[0]);

                        Mat result;
                        merge(channels,ycrcb);

                        cvtColor(ycrcb,result,CV_YCrCb2BGR);
                        
                        // FIXME: REMOVE THIS
                        Packet->frame.copyTo(Packet->processed_frame);
                        //result.copyTo(Packet->processed_frame);
                }
                
                // TODO: Normalize image
                ptr_queue[queue_id+1].Insert(Packet);
        }
        
        // Exit
        pthread_exit(0);
}

void* output (void *ptr) {
     
        clock_t frameRateClkStart = 0;
        clock_t frameRateClkEnd = 0;
        float numberOfFrames = 0;
        double FramesPerSecond = 0;
        
        // Receive which queue ID its supposed to access
        int queue_id = *((int *) ptr);
        
        namedWindow("Output Image",CV_WINDOW_NORMAL); //create a window

        while (1){
                Data* Packet = ptr_queue[queue_id].Remove();
                
                if (Packet->type == END){
                        delete Packet;
                        #if(DEBUG_ENABLED)
                                cout << "Received Valid = 0. Exiting " << queue_id << " stage\n";
                        #endif
                        break;                        
                }

                int minl = min (Packet->frame.rows, Packet->frame.cols);
                
                // Used so the thickness of the marks is based on the size
                // of the image
                int thickness = ceil((float) minl / 270.0);
                
                for (unsigned int i = 0; i < Packet->bounding_boxes.size(); i++) {
                        cv::rectangle(Packet->frame, 
                                Packet->bounding_boxes[i].p1, 
                                Packet->bounding_boxes[i].p2, 
                                cv::Scalar(255, 255, 255),
                                thickness);
                }
                for (unsigned int i = 0; i < Packet->landmarks.size(); i++) {
                        cv::circle(Packet->frame, 
                                Packet->landmarks[i].LE,
                                thickness,
                                cv::Scalar(255, 0, 0),
                                -1);
                        cv::circle(Packet->frame, 
                                Packet->landmarks[i].RE,
                                thickness,
                                cv::Scalar(255, 0, 0),
                                -1);
                        cv::circle(Packet->frame, 
                                Packet->landmarks[i].N,
                                thickness,
                                cv::Scalar(0, 255, 0),
                                -1);
                        cv::circle(Packet->frame, 
                                Packet->landmarks[i].LM,
                                thickness,
                                cv::Scalar(0, 0, 255),
                                -1);
                        cv::circle(Packet->frame, 
                                Packet->landmarks[i].RM,
                                thickness,
                                cv::Scalar(0, 0, 255),
                                -1);
                }
                if (Packet->type == VID){
                        // Show output
                        numberOfFrames++;
                        if (numberOfFrames==30){
                                frameRateClkEnd = clock();
                                double timePerFrame = (double)(frameRateClkEnd-frameRateClkStart) / CLOCKS_PER_SEC;
                                FramesPerSecond = (double)30/timePerFrame;
                                frameRateClkStart = frameRateClkEnd;
                                numberOfFrames = 0;
                        }
                        stringstream stream;
                        stream << "FPS: " << setprecision(4) << FramesPerSecond;
                        //<< "Res: " << Packet->frame.size().width << "x" 
                        //<< Packet->frame.size().height 
                                
                        string text = stream.str();
                        
                        // Add string to image
                        int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
                        double fontScale = 2;
                        int thickness = 3;
                        
                        int baseline=0;
                        Size textSize = getTextSize(text, fontFace,
                                                    fontScale, thickness, &baseline);
                        baseline += thickness;
                        
                        Point textOrg(0,textSize.height);
                        putText(Packet->frame, text, textOrg, fontFace, fontScale,
                        Scalar::all(255), thickness, 8);
                        
                        imshow("Output Image", Packet->frame);
                        resizeWindow("Output Image", 640, 480);
                        waitKey(1);
                        
                        delete Packet;
                } else if (Packet->type == IMG){
                        // Show output
                        //imshow("Output", Packet->frame);
                        cend = clock();
                        // Print Output
                        cout << "Execution time was: " << double(cend-cbegin) / CLOCKS_PER_SEC << endl;
        
                        stringstream ss;
                        ss << "outputs/output.jpg";// << file ;
                        string commS = ss.str();
                        
                        // remove input part
                        //string in = "inputs/";
                        //string::size_type i = commS.find(in);
                        //if (i!= std::string::npos) commS.erase(i,in.length());
                        const char* comm = commS.c_str();
                        cout << "writing " << comm << endl;
                        cv::imwrite(comm, Packet->frame);
                        
                        // Open window with detected objects
                        cv::imshow("Output Image", Packet->frame);
                        resizeWindow("Output Image", 640, 480);
                        cv::waitKey();                        
                        
                        delete Packet;
                }
        }
        
        // Exit
        destroyWindow("Output");
        pthread_exit(0);
}