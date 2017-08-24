
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
