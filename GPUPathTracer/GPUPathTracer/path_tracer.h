#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <cuda_runtime.h>
#include<vector>

struct Image;
struct Sphere;
struct Ray;
struct Camera;
struct Poly;
class PathTracer {

private:

	Image* image;

	int numSpheres;
	int numPolys;
	Sphere* spheres;
	void createDeviceData();
	void deleteDeviceData();

	void setUpScene();
	void loadEnvironmentMap();
	void FirstSetTexture(unsigned int pTexture,int &nCount);
public:
	PathTracer(Camera* cam);
	~PathTracer();

	void reset();

	Image* render();

	Camera* renderCamera;

	void prepMesh();
};

#endif // PATH_TRACER_H