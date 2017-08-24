
#include "../include/include.h"
#include "../include/def.h"
#include "../include/data.h"
#include "../include/queue.h"
#include "../include/auxiliar.h"
#include "../include/pthreads.h"

using namespace cv;
using namespace std;

CONF config;

double fps;

// Shared string with output thread to print information
string output_string;

void usage() {
	fprintf(stderr,"\n");
	fprintf(stderr,"Usage:  face-detector [options]\n");

	fprintf(stderr, "\nGeneral Options:\n");
	fprintf(stderr, "    --help,   -h      print this message\n");
	fprintf(stderr, "    --verbose -e       basic verbosity level\n");
	fprintf(stderr, "    --debug           enhanced verbosity level\n");

	fprintf(stderr, "\nExclusive Options: \n");
	fprintf(stderr, "    --image,    -i    <file_name>: Process image file\n");
	fprintf(stderr, "    --video,    -v    <file_name>: Process video file\n");
	fprintf(stderr, "    --database, -d    <file_name> <image_directory>");
	fprintf(stderr, ": Process a database with multiple images.\n");
	fprintf(stderr, "\t<file_name>: File with the name of the images (no extension, jpg only).\n");
	fprintf(stderr, "\t<image_directory>: Relative path to where the images are stored.\n");
	fprintf(stderr, "    --cam,      -c    <#>        : Process video from camera\n");
	fprintf(stderr, "\t<#>: Indicates which camera (if only 1 present, use 0) \n");

	fprintf(stderr, "\nAdditional Options:\n");
	fprintf(stderr, "    --nodisp,   -nd    Don't display image/video\n");
	fprintf(stderr, "    --record,   -r     Record video.\n");
	fprintf(stderr, "    --snapshot, -s     Save image(s).\n");
	fprintf(stderr, "    --log,      -l     Write log with performance results.\n");
	fprintf(stderr, "    --fddb,     -f     Write log with results based on FDDB database.\n");
	fprintf(stderr, "    --output,   -o    <folder_name>:  Store outputs in this directory.\n");
}

void init_conf() {
	config.type          = ILL;
	config.show_video    = true;
	config.record_video  = false;
	config.take_snapshot = false;
	config.verbose       = false;
	config.debug         = false;
	config.log_results   = false;
	config.fddb_results  = false;
	config.cam_id        = 0;
  config.short_file_name = "CAM";
}

int parse_arguments(int argc, char** argv) {

  int i = 1;
	while ( i < argc ) {
		if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) {
			usage();
			return 0;
		}
		else if ( strcmp(argv[i], "-e") == 0 ||
							strcmp(argv[i], "--verbose") == 0 ) {
			config.verbose = true;
		}
		else if ( strcmp(argv[i], "--debug")   == 0 ) {
			config.debug = true;
		}
		else if ( strcmp(argv[i], "-i") == 0 ||
							strcmp(argv[i], "--image") == 0 ) {
			++i;
			config.full_file_name = argv[i];
			if (config.type == ILL)
      	config.type = IMG;
			// Due to OS bug
      config.take_snapshot = true;
			config.show_video    = false;
		}
		else if ( strcmp(argv[i], "-d") == 0 ||
							strcmp(argv[i], "--database") == 0 ) {
			++i;
			config.full_file_name = argv[i];
			++i;
			config.image_dir = argv[i];
			config.short_file_name = argv[i];
			if (config.type == ILL)
      	config.type = DTB;
			// Due to OS bug
			config.show_video    = false;
		}
		else if ( strcmp(argv[i], "-v") == 0 ||
							strcmp(argv[i], "--video") == 0 ) {
			++i;
			config.full_file_name = argv[i];
			if (config.type == ILL)
      	config.type = VID;
		}
		else if ( strcmp(argv[i], "-c") == 0 ||
							strcmp(argv[i], "--cam") == 0 ) {
			++i;
			config.cam_id = atoi(argv[i]);
			if (config.type == ILL)
      	config.type = CAM;
		}
		else if ( strcmp(argv[i], "-nd") == 0 ||
							strcmp(argv[i], "--nodisp") == 0 ) {
			config.show_video = false;
		}
		else if ( strcmp(argv[i], "-r") == 0 ||
							strcmp(argv[i], "--record") == 0 ) {
			config.record_video = true;
		}
		else if ( strcmp(argv[i], "-s") == 0 ||
							strcmp(argv[i], "--snapshot") == 0 ) {
			config.take_snapshot = true;
		}
		else if ( strcmp(argv[i], "-l") == 0 ||
							strcmp(argv[i], "--log") == 0 ) {
			config.log_results = true;
		}
		else if ( strcmp(argv[i], "-f") == 0 ||
							strcmp(argv[i], "--fddb") == 0 ) {
			config.fddb_results = true;
		}
		else if ( strcmp(argv[i], "-o") == 0 ||
							strcmp(argv[i], "--output") == 0 ) {
			++i;
			config.output_dir = argv[i];
			// create folder if it doesn't exist
			std::string line;

			line.append("mkdir -p ");
			line.append(config.output_dir);
			cout << "command: " << line.c_str() << endl;
			if(system (line.c_str()) == -1){
				cout << "Couldn't create output directory\n";
				return 0;
			}
		}
		++i;

	}

	if (config.type == VID || config.type == IMG ){
	  const std::string text = config.full_file_name;
	  const std::regex slashrm(".*/");
	  std::stringstream result;
	  std::regex_replace(std::ostream_iterator<char>(result), text.begin(), text.end(), slashrm, "");
	  config.short_file_name = result.str();
	  config.short_file_name = config.short_file_name.substr(0, config.short_file_name.find_last_of("."));
	}

	if (config.output_dir == NULL){
		std::string temp = "outputs/";
		config.output_dir = new char[temp.length() + 1];
		strcpy(config.output_dir,temp.c_str());
		// TODO: Delete output_dir var
	}

	if (config.type == ILL){
		usage();
		return 0;
	}
	return 1;
}

void print_conf() {
        fprintf(stderr, "\n--------------");
        fprintf(stderr, "\nCONFIGURATION:\n");
        fprintf(stderr, "- Processing Type %d\n", config.type);
        fprintf(stderr, "- Show Video:     %d\n", config.show_video);
        fprintf(stderr, "- Record Video:   %d\n", config.record_video);
        fprintf(stderr, "- Take Snapshot:  %d\n", config.take_snapshot);
        fprintf(stderr, "- Write Log:      %d\n", config.log_results);
        fprintf(stderr, "- Write FDDB Log: %d\n", config.fddb_results);
        fprintf(stderr, "- File Name:      %s\n", config.short_file_name.c_str());
        fprintf(stderr, "- Output Dir:     %s\n", config.output_dir);
        if (config.verbose && config.debug)
					fprintf(stderr, "- verbose mode\n");
        if (config.debug)
					fprintf(stderr, "- debug mode\n");
        fprintf(stderr, "--------------\n");
}

int main(int argc, char* argv[]) {

	cv::VideoCapture video;
	cv::Mat img;

  // Timer
  double start, finish;
	double total_start;
	double total_start_with_setup = CLOCK();

	// Parse arguments
	init_conf();
	if ( !parse_arguments(argc, argv) ) return 0;
	if (config.verbose) print_conf();

	// Create Pthreads
	pthread_t* pthreads = new pthread_t [STAGE_COUNT];
	std::vector<int> pthread_id(STAGE_COUNT);
	for (int i = 0; i < STAGE_COUNT; i++){
	        pthread_id[i] = i;
	}

	pthread_create(&pthreads[0], NULL, preprocess   , (void *)&pthread_id[0]);
	pthread_create(&pthreads[1], NULL, pnet         , (void *)&pthread_id[1]);
	pthread_create(&pthreads[2], NULL, rnet         , (void *)&pthread_id[2]);
	pthread_create(&pthreads[3], NULL, onet         , (void *)&pthread_id[3]);
	pthread_create(&pthreads[STAGE_COUNT-2], NULL, postprocess  , (void *)&pthread_id[STAGE_COUNT-2]);
	pthread_create(&pthreads[STAGE_COUNT-1], NULL, output       , (void *)&pthread_id[STAGE_COUNT-1]);

	//Wait for all stages to setup
	Data* StartupPacket = new Data;
	StartupPacket->type = STU;
	for (int i = 0; i < STAGE_COUNT; i++){
		ptr_queue[i].Insert(StartupPacket);
	}

	// Wait for children to finish setting up
	StartupPacket->WaitForCounter(STAGE_COUNT);

	delete StartupPacket;

	total_start = CLOCK();

	switch (config.type){
	  case CAM:
	  case VID: {

			// Start ncurses lib
			// Needed to process input keyboard correctly
			initscr();

      // open the video file for reading
			if (config.type == CAM){
				video.open(config.cam_id);
			} else {
				video.open(config.full_file_name);
			}

			// If not successfull, exit program
      if (!video.isOpened()) {
	      endwin();
	      cout << "Cannot open the video" << endl;
				Data* Packet = new Data;
	      Packet->type = END;
	      ptr_queue[0].Insert(Packet);
	      break;
      }

			if (config.type == CAM){
				fps = CAM_FPS;
				video.set(CV_CAP_PROP_FPS, fps);
				video.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
				video.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);
			} else {
				fps = video.get(CV_CAP_PROP_FPS);
			}

      double dWidth   = video.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
      double dHeight  = video.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

      bool finished = 0;
      while (!finished){

				// Wait for a little while (not full fps) before grabing next frame
				if(config.type == VID)
					timeout (950/fps);
				else
					timeout (800/fps);

	      // Control screen
	      clear();
	      printw("Heterogeneous Framework for Face Detection \nby Julian Gutierrez\n");
	      printw("------------------------------------------\n");

	      if(config.verbose || config.debug){
          printw("Frame size        : %.2f x %.2f\n", dWidth, dHeight);
          printw("Frame rate        : %.2f\n", fps);
					printw("Current Runtime   : %.3f s\n", (CLOCK() - total_start_with_setup)/1000);
	      }

	      printw("[%d] Show Video     : Press 'v' to change.\n", (config.show_video)?1:0);
	      printw("[%d] Record Video   : Press 'r' to toggle recording.\n", (config.record_video)?1:0);
				printw("[%d] Take Snapshot  : Press 's' to take snapshot.\n", (config.take_snapshot)?1:0);
				printw("[%d] Write Log File : Press 'l' to toggle writing.\n", (config.log_results)?1:0);
	      printw("---------------------\n");
				if (!output_string.empty() && (config.verbose || config.debug))
					printw("%s", output_string.c_str());
				printw("Press 'q' to quit.\n");

				// Read packet
	      Data* Packet = new Data;

				// Record time
				Packet->start_time = CLOCK();
				start = CLOCK();

	      bool bSuccess = video.read(Packet->frame); // read a new frame from video

	      if (!bSuccess) {//if not success, break loop
          printw("Cannot read a frame from video stream\n");
          Packet->type = END;
          ptr_queue[0].Insert(Packet);
          break;
	      }

	      Packet->type = VID;

				// Record time
				finish = CLOCK();
				Packet->stage_time[STAGE_COUNT] = finish - start;

	      ptr_queue[0].Insert(Packet);

	      int c = getch();
	      switch (c){
          case 113: { // 'q'
            // Quit
            Data* FinishPacket = new Data;
            FinishPacket->type = END;
            ptr_queue[0].Insert(FinishPacket);
            finished = 1;
            break;
          }
          case 114: { // 'r'
            // Record Video
            config.record_video = !config.record_video;
            break;
          }
          case 115: { // 's'
            // Take Snapshot
            config.take_snapshot = !config.take_snapshot;
            break;
          }
					case 108: { // 'l'
            // Take Snapshot
            config.log_results = !config.log_results;
            break;
          }
          case 118: { // 'v'
            // Show Video
            config.show_video = !config.show_video;
            break;
          }
          default: {
          }
	      }
      }

      video.release();

      break;
	  }
	  case IMG: {

			Data* Packet = new Data;

			// Record time
			Packet->start_time = CLOCK();
			start = CLOCK();

      // open image file for reading
			img = cv::imread(config.full_file_name, -1);
			if(img.empty()){
				//endwin();
				cout << "Unable to decode image " << config.full_file_name << endl;
			  Packet->type = END;
	      ptr_queue[0].Insert(Packet);
	      break;
			}

      img.copyTo(Packet->frame);
      Packet->type = IMG;

			// Record time
			finish = CLOCK();
			Packet->stage_time[STAGE_COUNT] = finish - start;

			ptr_queue[0].Insert(Packet);

      // Finish program
      Data* FinishPacket = new Data;
      FinishPacket->type = END;
			ptr_queue[0].Insert(FinishPacket);

      break;
	  }
		case DTB: {

			// Setup Name of files to be processed
			std::ifstream inputFile(config.full_file_name);

			std::vector<std::string> fileList;

			if(!inputFile) {
				Data* Packet = new Data;
		    cout << "Unable to read file correctly " << config.full_file_name << endl;
			  Packet->type = END;
	      ptr_queue[0].Insert(Packet);
		    return 1;
		  }

			std::string line;
			while(std::getline(inputFile, line)) {
		    fileList.push_back(line);
		  }

			std::cout << "Number of images to be analyzed: " << fileList.size() << "\n";

			std::vector<std::string>::const_iterator it(fileList.begin());
		  std::vector<std::string>::const_iterator end(fileList.end());
		  for(;it != end; ++it) {

				std::string line;

				line.append(config.image_dir);
				//line.append("/");
				line.append(it->c_str());
				line.append(".jpg");

				Data* Packet = new Data;

				Packet->name = it->c_str();

				// Record time
				Packet->start_time = CLOCK();
				start = CLOCK();

	      // open image file for reading
				img = cv::imread(line.c_str(), -1);

				if(img.empty()){
					//endwin();
					cout << "Unable to decode image " << line.c_str() << endl;
				  Packet->type = END;
		      ptr_queue[0].Insert(Packet);
		      break;
				}

	      img.copyTo(Packet->frame);
	      Packet->type = DTB;

				// Record time
				finish = CLOCK();
				Packet->stage_time[STAGE_COUNT] = finish - start;

				ptr_queue[0].Insert(Packet);
		  }

      // Finish program
      Data* FinishPacket = new Data;
      FinishPacket->type = END;
			ptr_queue[0].Insert(FinishPacket);

      break;
	  }
	  default: {
      // Nothing
      break;
	  }
	}

	// Close ncurses
	if (config.type == CAM || config.type == VID)
		endwin();

	// Wait for children
	if(config.debug) cout << "Waiting for child threads to exit successfully.\n";
	for (int i = 0; i < STAGE_COUNT; i++){
		pthread_join(pthreads[i], NULL);
	}

	cout << "Total Application Runtime" << endl
			 << "\tIncluding Setup Time  :  " << CLOCK() - total_start_with_setup << " ms" << endl
			 << "\tWithout Setup Time    :  " << CLOCK() - total_start << " ms" << endl << endl;

	// Done
	return 0;
}
