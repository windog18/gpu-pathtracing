#ifndef PATH_TRACER_KERNEL_H
#define PATH_TRACER_KERNEL_H

#include"geometry/triangle.h"
#include <vector>
// Necessary forward declaration:
extern "C"
	void launchKernel(int numPixels,Color* pixels ,Material *device_materialList, int counter, Camera renderCamera);

extern "C"
void constructKDTree(std::vector<Triangle> &triarr, float minx, float miny, float minz, float maxx, float maxy, float maxz);

#endif // PATH_TRACER_KERNEL_H