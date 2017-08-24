#ifndef PNET_H
#define PNET_H

#include "bnet.h"

using namespace std;
using namespace caffe;

using std::string;

class PNet : public BNet{
	public:
		PNet(const string& model_file,
				 const string& trained_file) : BNet(model_file, trained_file){}

		void RetrieveOutput     (std::vector<int>& shape, std::vector< std::vector <float>>& data);

};

#endif
