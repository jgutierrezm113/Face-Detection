
#include "../include/auxiliar.h"

using namespace std;
using namespace caffe;

using std::string;

double _fpsstart;
double _fps1sec;

// Function to return indices of sorted array
vector<int> ordered(vector<float> values) {
  std::vector<int> indices(values.size());
  std::size_t n(0);
  std::generate(std::begin(indices), std::end(indices), [&]{ return n++; });

  std::sort(
    std::begin(indices), std::end(indices),
    [&](size_t a, size_t b) { return values[a] < values[b]; }
  );
  return indices;
}

vector<int> nms (std::vector <BBox>total_boxes,
                float threshold,
                bool  type){

  vector <int> pick;

  if (total_boxes.size() == 0){
    return pick;
  }

  vector <float> x1  (total_boxes.size());
  vector <float> y1  (total_boxes.size());
  vector <float> x2  (total_boxes.size());
  vector <float> y2  (total_boxes.size());
  vector <float> s   (total_boxes.size());
  vector <float> area(total_boxes.size());

  // Initialize vectors
  for (unsigned int i = 0; i < total_boxes.size(); i++){
    x1[i] = total_boxes[i].p1.x;
    y1[i] = total_boxes[i].p1.y;
    x2[i] = total_boxes[i].p2.x;
    y2[i] = total_boxes[i].p2.y;
     s[i] = total_boxes[i].score;
    area[i] = ((float)x2[i]-(float)x1[i]) * ((float)y2[i]-(float)y1[i]);
  }

  // Sort s and create indexes
  vector <int> I = ordered(s);

  while (I.size() > 0){

    // To store new Indexes
    vector <int> Inew;

    int i = I[I.size() - 1];
    pick.push_back(i);

    for (unsigned int j = 0; j < I.size()-1; j++){
      float   xx1 = max(x1[i],  x1[I[j]]);
      float   yy1 = max(y1[i],  y1[I[j]]);
      float   xx2 = min(x2[i],  x2[I[j]]);
      float   yy2 = min(y2[i],  y2[I[j]]);
      float     w = max(  0.0f, (xx2-xx1));
      float     h = max(  0.0f, (yy2-yy1));
      float inter = w * h;
      float   out;
      if (type == false){ // Union
        out = inter/(area[i] + area[I[j]] - inter);
      } else { // Min
        out = inter/min(area[i], area[I[j]]);
      }
      // Add index to Inew if under threshold
      if (out <= threshold){
        Inew.push_back(I[j]);
      }
    }
    // Copy new I into I
    I.swap(Inew);
    Inew.clear();
  }
  return pick;

}

vector<BBox> generateBoundingBox(std::vector< std::vector <float>> data,
                                std::vector<int> shape_map,
                                float scale,
                                float threshold){
  int stride   = 2;
  int cellsize = 12;

  vector<BBox> temp_boxes;
  for (int y = 0; y < shape_map[2]; y++){
    for (int x = 0; x < shape_map[3]; x++){
      // We need to access the second array.
      if (data[1][(shape_map[2] + y) * shape_map[3] + x] >= threshold){
        BBox temp_box;

        // Points for Bounding Boxes
        cv::Point p1(floor((stride*x+1)/scale),
                     floor((stride*y+1)/scale));
        cv::Point p2(floor((stride*x+cellsize-1+1)/scale),
                     floor((stride*y+cellsize-1+1)/scale));

        temp_box.p1 = p1;
        temp_box.p2 = p2;

        // Score
        temp_box.score = data[1][(shape_map[2] + y) * shape_map[3] + x];

        // Reg (dx1,dy1,dx2,dy2)
        cv::Point dp1 (data[0][(0*shape_map[2] + y) * shape_map[3] + x],
                       data[0][(1*shape_map[2] + y) * shape_map[3] + x]);
        cv::Point dp2 (data[0][(2*shape_map[2] + y) * shape_map[3] + x],
                       data[0][(3*shape_map[2] + y) * shape_map[3] + x]);

        temp_box.dP1 = dp1;
        temp_box.dP2 = dp2;

        // Add box to bounding boxes
        temp_boxes.push_back(temp_box);
      }
    }
  }
  return temp_boxes;
}

void padBoundingBox(std::vector <BBox> &bounding_boxes, int imgHeight, int imgWidth){
  for (unsigned int j = 0; j < bounding_boxes.size(); j++){
    if (bounding_boxes[j].p2.x >= imgWidth){ // P2.x > w
      // shift box
      bounding_boxes[j].p1.x -= bounding_boxes[j].p2.x - imgWidth;
      bounding_boxes[j].p2.x = imgWidth - 1;
    }

    if (bounding_boxes[j].p2.y >= imgHeight){ // P2.y > h
      // shift box
      bounding_boxes[j].p1.y -= bounding_boxes[j].p2.y - imgHeight;
      bounding_boxes[j].p2.y = imgHeight - 1;
    }

    if (bounding_boxes[j].p1.x < 0){
      // shift box
      bounding_boxes[j].p2.x -= bounding_boxes[j].p1.x;
      bounding_boxes[j].p1.x = 0;
    }

    if (bounding_boxes[j].p1.y < 0){
      // shift box
      bounding_boxes[j].p2.y -= bounding_boxes[j].p1.y;
      bounding_boxes[j].p1.y = 0;
    }
  }
}

void writeOutputImage(Data* Packet) {
  int minl = min (Packet->frame.rows, Packet->frame.cols);

  // Used so the thickness of the marks is based on the size
  // of the image
  int thickness = ceil((float) minl / 270.0);

  for (unsigned int i = 0; i < Packet->bounding_boxes.size(); i++) {
    cv::rectangle(Packet->frame,
                  Packet->bounding_boxes[i].p1,
                  Packet->bounding_boxes[i].p2,
                  cv::Scalar(255, 255, 255),
                  thickness);
  }
  for (unsigned int i = 0; i < Packet->landmarks.size(); i++) {
    cv::circle(Packet->frame,
              Packet->landmarks[i].LE,
              thickness,
              cv::Scalar(255, 0, 0),
              -1);
    cv::circle(Packet->frame,
              Packet->landmarks[i].RE,
              thickness,
              cv::Scalar(255, 0, 0),
              -1);
    cv::circle(Packet->frame,
              Packet->landmarks[i].N,
              thickness,
              cv::Scalar(0, 255, 0),
              -1);
    cv::circle(Packet->frame,
              Packet->landmarks[i].LM,
              thickness,
              cv::Scalar(0, 0, 255),
              -1);
    cv::circle(Packet->frame,
              Packet->landmarks[i].RM,
              thickness,
              cv::Scalar(0, 0, 255),
              -1);
  }
}

// Timing functions
double CLOCK() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC,  &t);
  return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

void avginit(){
  _fpsstart = 0;
  _fps1sec  = 0;
}

double avgfps(double _avgfps) {
  if(CLOCK() - _fpsstart > 1000) {
    _fpsstart = CLOCK();
    _avgfps = 0.7*_avgfps+0.3*_fps1sec;
    _fps1sec = 0;
  }
  _fps1sec++;
  return _avgfps;
}

double avgdur(double newdur, double _avgdur) {
  _avgdur=0.98*_avgdur+0.02*newdur;
  return _avgdur;
}
