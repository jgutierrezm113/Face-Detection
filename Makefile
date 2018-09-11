GCC := g++

RM = rm -f

# Specify Caffe Instalation
CaffeLocation = /home/julian/caffe
CaffeLIB = -L$(CaffeLocation)/build/lib
CaffeINC = -I$(CaffeLocation)/include/

# Specify Cuda Instalation
CudaLIB = -L/usr/local/cuda/lib
CudaINC = -I/usr/local/cuda/include/
#
# Specify opencv Installation
#opencvLocation = /usr/local/opencv
opencvLIB= -L/usr/lib
opencvINC = -I/usr/include

NetLocation = ./net

NetLIB = -L$(NetLocation)

# g++ main.cpp queue.cpp -lpthread -lopencv_core -lopencv_imgproc -lopencv_highgui -lncurses
GccFLAGS =  -pthread -std=c++11 -O3
GccLib = $(CaffeLIB) $(NetLIB) $(opencvLIB) $(CudaLIB) 
GccInc = $(CaffeINC) $(opencvINC) $(CudaINC)

GccLinkFLAGS = -lpthread -lprotobuf -lglog `pkg-config opencv --cflags --libs` -lboost_system -lcaffe -lnet -lncurses

debug: GccFLAGS += -DDEBUG -g -Wall
debug: all

# Doing only one compilation, for speed purposes
Dependencies = src/pthreads.o src/auxiliar.o src/pnet_threads.o src/rnet_threads.o src/onet_threads.o src/data.o

# The build target executable
TARGET = face_detector

all: build

build: $(TARGET)

# Create executable
$(TARGET): src/main.cpp $(Dependencies) include/def.h net/libnet.so
	$(GCC) $(GccLib) $(GccInc) -Wl,-rpath=$(NetLocation) $< $(Dependencies) -o $@ $(GccFLAGS) $(GccLinkFLAGS)

# Create Shared library for net objects
net/libnet.so: net/bnet.o net/pnet.o net/rnet.o net/onet.o
	$(GCC) $(CaffeINC) $(GccFLAGS) -shared $< net/pnet.o net/rnet.o net/onet.o -o $@

net/%.o: net/%.cpp
	$(GCC) $(GccInc) $(GccFLAGS) -c -fpic $< -o $@

src/%.o: src/%.cpp
	$(GCC) $(GccInc) $(GccFLAGS) -c -fpic $< -o $@

clean:
	$(RM) $(TARGET) *.o net/*.so net/*.o src/*.o *.tar* *.core*
