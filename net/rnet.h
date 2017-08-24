#ifndef RNET_H
#define RNET_H

#include "bnet.h"

using namespace std;
using namespace caffe;

using std::string;

class RNet : public BNet{
  public:
        RNet(const string& model_file,
             const string& trained_file) : BNet(model_file, trained_file){}
  
        void RetrieveOutput     (std::vector<int>& shape, std::vector< std::vector <float>>& data);

};

#endif