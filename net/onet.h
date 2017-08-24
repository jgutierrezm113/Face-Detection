#ifndef ONET_H
#define ONET_H

#include "bnet.h"

using namespace std;
using namespace caffe;

using std::string;

class ONet : public BNet{
	public:
		ONet(const string& model_file,
				 const string& trained_file) : BNet(model_file, trained_file){}

		void RetrieveOutput     (std::vector<int>& shape, std::vector< std::vector <float>>& data);
};

#endif
