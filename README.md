# RTFD
Real Time Face Detection

This is a project mixing Caffe and C++ using an already designed approach from paper "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks" for face and landmark detection on real time videos.

# CUDA
https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73

# Instalation
http://caffe.berkeleyvision.org/install_apt.html
http://caffe.berkeleyvision.org/installation.html#compilation

https://github.com/BVLC/caffe/wiki/Commonly-encountered-build-issues

protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto
