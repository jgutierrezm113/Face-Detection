
#include "bnet.h"

using namespace std;
using namespace caffe;

using std::string;

/*
This function should be implemented for each derived class accordingly
void RetrieveOutput     (const& std::vector<int> shape, std::vector< std::vector <float>>& data);
*/

BNet::BNet(const string& model_file, const string& trained_file) {

        // Load the network
        net.reset(new Net<float>(model_file, TEST));
        net->CopyTrainedLayersFrom(trained_file);

}

void BNet::SetInputGeometry (cv::Size input){
        input_geometry = input;
}

void BNet::FeedInput (std::vector<cv::Mat>& imgs){
        
        // Make sure input_geometry is same size as image
        for (unsigned int it = 0; it < imgs.size(); it++){
                CHECK_EQ(input_geometry.height, imgs.at(it).rows) << 
                "Image height dimention is different than specified input.";
                CHECK_EQ(input_geometry.width, imgs.at(it).cols) << 
                "Image width dimention is different than specified input.";          
        }
       
        // Function Call
        Blob<float>* input_layer = net->input_blobs()[0];
        
        input_layer->Reshape(imgs.size(), 
                                3,
                                input_geometry.height, 
                                input_geometry.width);
                       
        // Forward dimension change to all layers.
        net->Reshape();

        vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels);

        PreProcess(&input_channels, &imgs);
}

void BNet::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
        Blob<float>* input_layer = net->input_blobs()[0];

        const std::vector<int> shape = input_layer->shape();
              
        float* input_data = input_layer->mutable_cpu_data();
        for (int i = 0; i < shape[0]*shape[1]; ++i) { // num * channels (boxes*3)
                cv::Size size(shape[3],shape[2]); 
                cv::Mat channel(size, CV_32FC1, input_data);
                input_channels->push_back(channel);
                input_data += shape[3] * shape[2]; // width * hight
        }
}

void BNet::PreProcess(std::vector<cv::Mat>* input_channels, // will be 3 times bigger than imgs
                      std::vector<cv::Mat>* imgs){
        
        vector<cv::Mat>* input_channels_org = input_channels;
        Blob<float>* input_layer = net->input_blobs()[0];

        const std::vector<int> shape = input_layer->shape();
        
        vector<cv::Mat> sample_float (imgs->size());

        for (unsigned int i = 0; i < imgs->size(); i++){
                cv::split(imgs->at(i), &input_channels->at(i*3));
        }
        
        CHECK(reinterpret_cast<float*>(input_channels_org->at(0).data)
        == net->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
        
}

void BNet::Forward (void){
        net->Forward();
}

std::shared_ptr<Net<float> > BNet::GetNet (void){
        return net;
}
