#ifndef BNET_H
#define BNET_H

#include "../include/def.h"

using namespace std;
using namespace caffe;

using std::string;

class BNet {
  public:
        BNet(const string& model_file,
             const string& trained_file);
  
        std::shared_ptr<Net<float> > GetNet (void);
        void SetInputGeometry   (cv::Size input);
        void FeedInput          (std::vector<cv::Mat>& imgs);
        void Forward            (void);
        //void RetrieveOutput     (const& std::vector<int> shape, std::vector< std::vector <float>>& data);

  private:        
        void WrapInputLayer         (std::vector<cv::Mat>* input_channels);
        void PreProcess             (std::vector<cv::Mat>* input_channels,
                                     std::vector<cv::Mat>* imgs);

        std::shared_ptr<Net<float> > net;
        
        cv::Size input_geometry;
};

#endif