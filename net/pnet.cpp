
#include "pnet.h"

using namespace std;
using namespace caffe;

using std::string;

void PNet::RetrieveOutput (std::vector<int>& shape, std::vector< std::vector <float>>& data){
        
        Blob<float>* output_layer_reg = GetNet()->output_blobs()[0];
        Blob<float>* output_layer_map = GetNet()->output_blobs()[1];
        
        const std::vector<int> shape_reg = output_layer_reg->shape();
        const std::vector<int> shape_map = output_layer_map->shape();

        // Redirect shape pointer 
        shape.push_back(shape_reg[0]);
        shape.push_back(shape_reg[1]);
        shape.push_back(shape_reg[2]);
        shape.push_back(shape_reg[3]);
        
        // Write output
        const float* begin = output_layer_reg->cpu_data();
        const float*   end = begin + shape_reg[0]*shape_reg[1]*shape_reg[2]*shape_reg[3];
        
        vector<float> output_data_reg(begin,end);
        
        data.push_back(output_data_reg);
                        
        const float* mbegin = output_layer_map->cpu_data();
        const float*  mend = mbegin + shape_map[0]*shape_map[1]*shape_map[2]*shape_map[3];
        
        vector<float> output_data_map(mbegin,mend);
        
        data.push_back(output_data_map);
}