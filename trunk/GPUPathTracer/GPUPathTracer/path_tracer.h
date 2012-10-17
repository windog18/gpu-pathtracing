#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <cuda_runtime.h>
#include<vector>
#include<string>
#include "material.h"
#include "hdrloader.h"
struct Image;
struct Sphere;
struct Ray;
struct Camera;
struct Poly;
class obj;
class PathTracer {

private:

	Image* image;
	Material *device_materialList;

	int numSpheres;
	int numPolys;
	Sphere* spheres;
	void createDeviceData();
	void deleteDeviceData();

	void setUpScene();
	void setEnvironmentMap(std::string filePath);
	void loadSubImage(int fromX,int fromY,int toX,int toY,const HDRLoaderResult& data,Image *store);
	void FirstSetTexture(unsigned int pTexture,int nCount);
public:
	PathTracer(Camera* cam);
	~PathTracer();

	void reset();

	Image* render();

	Camera* renderCamera;

	void prepMesh();
};

#endif // PATH_TRACER_H