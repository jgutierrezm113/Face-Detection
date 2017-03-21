
#include "../include/include.h"
#include "../include/def.h"
#include "../include/data.h"
#include "../include/queue.h"
#include "../include/auxiliar.h"
#include "../include/pthreads.h"

using namespace cv;
using namespace std;

extern Queue ptr_queue[STAGE_COUNT];

char *file_name;

clock_t cbegin;
clock_t cend;

int main(int argc, char* argv[]) {
        
        Processing_Type type;
        VideoCapture video;
        cv::Mat img;

        // Process Inputs
        if (argc == 1){
                // Use web-cam
                type = CAM;
                //FIXME: Not yet implemented
                exit(0);
                video.open(0); // open the video file for reading
        } else if (argc == 3){
                std::string arg = argv[1];
                if (arg == "-v"){
                        // Processing video file
                        type = VID;
                        // open the video file for reading
                        file_name = argv[2];
                        video.open(argv[2]);
                } else if (arg == "-i"){
                        // Processing image file
                        type = IMG;
                        file_name = argv[2];
                        img = cv::imread(file_name, -1);
                        CHECK(!img.empty()) << "Unable to decode image " << argv[2];
        
                } else {
                        cout << "Invalid flag used, please try again.\n";
                        exit (-1); 
                }
        } else {
                cout << "Invalid arguments used, please try again.\n";
                exit (-1);
        }
        
        // TODO: Can do this better
        // Create PNET threads
        pthread_t* pthreads = new pthread_t [STAGE_COUNT];
        int pthread_id[STAGE_COUNT] = {0, 1, 2, 3, 4};
        
        pthread_create(&pthreads[0], NULL, preprocess   , (void *)&pthread_id[0]);
        pthread_create(&pthreads[1], NULL, pnet         , (void *)&pthread_id[1]);
        pthread_create(&pthreads[2], NULL, rnet         , (void *)&pthread_id[2]);
        pthread_create(&pthreads[3], NULL, onet         , (void *)&pthread_id[3]);
        pthread_create(&pthreads[4], NULL, output       , (void *)&pthread_id[4]);
              
        switch (type){
                
                case VID: {
                        
                        if (!video.isOpened()) { // if not success, exit program
                                endwin();
                                cout << "Cannot open the video" << endl;
                                exit(-1);
                        }
                        
                        // Start
                        initscr(); // Needed to process input keyboard correctly
                        printw("Face Detection!\n");
                        printw("Press 'q' to quit:\n");

                        double fps      = video.get(CV_CAP_PROP_FPS);
                        timeout (1000/fps);
                        
                        #if(DEBUG_ENABLED)
                                double dWidth   = video.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
                                double dHeight  = video.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
                        
                                printw("Frame size : %lf x %lf\n", dWidth, dHeight);
                                printw("Frame rate : %lf\n", fps);
                        #endif 
                        
                        while (1){
                                Data* Packet = new Data;

                                bool bSuccess = video.read(Packet->frame); // read a new frame from video

                                if (!bSuccess) {//if not success, break loop
                                        printw("Cannot read a frame from video stream\n");
                                        Packet->type = END;
                                        ptr_queue[0].Insert(Packet);
                                        break;
                                }

                                Packet->type = VID;
                                ptr_queue[0].Insert(Packet);
                                int c = getch();
                                if (c == 113){
                                        Data* FinishPacket = new Data;
                                        FinishPacket->type = END;
                                        ptr_queue[0].Insert(FinishPacket);
                                        break; 
                                }                                
                        }
                        
                        endwin();
                        video.release(); 
        
                        break;
                }
                case IMG: {

                        cbegin = clock();
                        Data* Packet = new Data;

                        img.copyTo(Packet->frame);
                        Packet->type = IMG;
                        ptr_queue[0].Insert(Packet);
                        
                        // Finish program
                        Data* FinishPacket = new Data;
                        FinishPacket->type = END;
                        ptr_queue[0].Insert(FinishPacket);
                        break; 
                }
                default: {

                }
        }

        // Wait for children
        for (int i = 0; i < STAGE_COUNT; i++){
                pthread_join(pthreads[i], NULL);
        }
        
        // Done
        return 0;
}
