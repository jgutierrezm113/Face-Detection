#ifndef AUXILIAR_H
#define AUXILIAR_H

#include "include.h"
#include "def.h"
#include "data.h"
#include "shared.h"

/*
 * Auxiliar functions used in process
 */

std::vector<int> ordered(std::vector<float> values);

std::vector<int> nms (std::vector <BBox>total_boxes,
																float threshold,
																bool  type);

std::vector<BBox> generateBoundingBox(std::vector< std::vector <float>> data,
																std::vector<int> shape_map,
																float scale,
																float threshold);

void padBoundingBox(std::vector <BBox> &bounding_boxes,
																int imgHeight,
																int imgWidth);

void writeOutputImage(Data* Packet);

double CLOCK();
void avginit();
double avgfps(double _avgfps);
double avgdur(double newdur, double _avgdur);

#endif
