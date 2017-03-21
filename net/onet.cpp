
#include "onet.h"

using namespace std;
using namespace caffe;

using std::string;

void ONet::RetrieveOutput (std::vector<int>& shape, std::vector< std::vector <float>>& data){
        
        Blob<float>* output_layer_score = GetNet()->output_blobs()[2];
        Blob<float>* output_layer_points = GetNet()->output_blobs()[1];
        Blob<float>* output_layer_mv = GetNet()->output_blobs()[0];
        
        const std::vector<int> shape_score = output_layer_score->shape();
        const std::vector<int> shape_points = output_layer_points->shape();
        const std::vector<int> shape_mv = output_layer_mv->shape();

        // Redirect shape pointer 
        shape.push_back(shape_score[0]);
        shape.push_back(shape_score[1]);
        
        // Write output
        const float* begin = output_layer_score->cpu_data();
        const float*   end = begin + shape_score[0]*shape_score[1];
        
        vector<float> output_data_score(begin,end);
        
        const float* pbegin = output_layer_points->cpu_data();
        const float*   pend = pbegin + shape_points[0]*shape_points[1];
        
        vector<float> output_data_points(pbegin,pend);
        
        const float* mbegin = output_layer_mv->cpu_data();
        const float*  mend = mbegin + shape_mv[0]*shape_mv[1];
        
        vector<float> output_data_mv(mbegin,mend);        
       
        data.push_back(output_data_mv);
        data.push_back(output_data_points);
        data.push_back(output_data_score);
    
}

void ONet::PreProcessMatlab(std::vector <BBox> bounding_boxes, const string& image_name){
          
        // Write File with box info
        cout << "- Writing Boxes to file" << endl;
        ofstream myfile;
        myfile.open ("matlab/boxes.txt");
        
        for (unsigned int i = 0; i < bounding_boxes.size(); i++) {
                myfile  << bounding_boxes[i].p1.x << " "
                        << bounding_boxes[i].p1.y << " "
                        << bounding_boxes[i].p2.x << " "
                        << bounding_boxes[i].p2.y << endl;
        }

        myfile.close();
       
        // Shape Net Input
        Blob<float>* input_layer = GetNet()->input_blobs()[0];
        
        input_layer->Reshape(bounding_boxes.size(), 
                                3,
                                48, 
                                48);
                       
        // Forward dimension change to all layers.
        GetNet()->Reshape();

        vector<cv::Mat> input_channels;
        
        const std::vector<int> shape = input_layer->shape();
              
        float* input_data = input_layer->mutable_cpu_data();
        for (int i = 0; i < shape[0]*shape[1]; ++i) { // num * channels (boxes*3)
                cv::Size size(shape[3],shape[2]); 
                cv::Mat channel(size, CV_32FC1, input_data);
                input_channels.push_back(channel);
                input_data += shape[3] * shape[2]; // width * hight
        }
        
        // Call matlab function to generate input
        stringstream cpImg;

        cpImg << "cp " << image_name << " matlab/img.jpg";
        string commS = cpImg.str();
        const char* comm = commS.c_str();
        
        cout << "- Running matlab engine" << endl;
        
        system (comm);
        putenv("LD_LIBRARY_PATH=/usr/local/MATLAB/R2016b/bin/glnxa64:/usr/local/MATLAB/R2016b/sys/os/glnxa64");
        putenv("LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfreetype.so.6");
        system ("cd matlab; ./onetInput; cd ..");
        putenv("LD_LIBRARY_PATH=");
        putenv("LD_PRELOAD=");
        
        // Read values from file
        cout << "- Reading data from file" << endl;
        ifstream myfile2; 
        myfile2.open ("matlab/imgout.txt");
        
        float* pointertodata = GetNet()->input_blobs()[0]->mutable_cpu_data();
        float tmp;
        int i = 0;
        while (myfile2 >> tmp){
                pointertodata[i] = tmp;
                i++;
        }
        myfile2.close();
        
        CHECK(reinterpret_cast<float*>(pointertodata)
        == GetNet()->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
        
        // delete files
        system ("rm matlab/*.txt matlab/img.jpg");
}