#include "path_tracer.h"

#include "image.h"
#include "sphere.h"
#include "camera.h"
#include "poly.h"

#include "windows_include.h"

// System:
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <iostream>

// CUDA:
#include <cuda_runtime.h>
#include "cutil_math.h"

#include "cuda_safe_call.h"
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>

#include "path_tracer_kernel.h"

#include "objcore/objloader.h"
#include "geometry/triangle.h"
//#include "cukd/primitives.h"
PathTracer::PathTracer(Camera* cam) {

	renderCamera = cam;

	image = newImage(renderCamera->resolution.x, renderCamera->resolution.y);

	numPolys = 0;

	prepMesh();

	setUpScene();

	createDeviceData();

	
}


PathTracer::~PathTracer() {

	deleteDeviceData();
	deleteImage(image);

}

void::PathTracer::prepMesh(){
	obj* m = new obj();
	objLoader meshload("../box.obj", m);

/*	polys = new Poly[m->getFaces()->size()];

	for(int i=0; i<m->getFaces()->size(); i++){
		glm::vec4 p0 = m->getPoints()->operator[](m->getFaces()->operator[](i)[0]);
		glm::vec4 p1 = m->getPoints()->operator[](m->getFaces()->operator[](i)[1]);
		glm::vec4 p2 = m->getPoints()->operator[](m->getFaces()->operator[](i)[2]);
		glm::vec4 n = m->getNormals()->operator[](m->getFaceNormals()->operator[](i)[0]);
		polys[i].n = make_float3(n[0], n[1], n[2]);
		polys[i].p0 = make_float3(p0[0], p0[1], p0[2]);
		polys[i].p1 = make_float3(p1[0], p1[1], p1[2]);
		polys[i].p2 = make_float3(p2[0], p2[1], p2[2]);
	}*/

	numPolys = m->getFaces()->size();

	std::vector<Triangle> tris;
	for(int i=0; i<numPolys; i++){
		Triangle tri;
		for(int j = 0; j < 3; ++j) {
			glm::vec4 p = m->getPoints()->operator[](m->getFaces()->operator[](i)[j]);
			glm::vec4 n = m->getNormals()->operator[](m->getFaceNormals()->operator[](i)[j]);
			if(j == 0){
				tri.v1.x = p[0];
				tri.v1.y = p[1];
				tri.v1.z = p[2];
				tri.n1.x = n[0];
				tri.n1.y = n[1];
				tri.n1.z = n[2];
			}else if(j == 1){
				tri.v2.x = p[0];
				tri.v2.y = p[1];
				tri.v2.z = p[2];
				tri.n2.x = n[0];
				tri.n2.y = n[1];
				tri.n2.z = n[2];
			}else{
				tri.v3.x = p[0];
				tri.v3.y = p[1];
				tri.v3.z = p[2];
				tri.n3.x = n[0];
				tri.n3.y = n[1];
				tri.n3.z = n[2];
			}
        }
		tris.push_back(tri);
	}
	constructKDTree(tris, m->getMin()[0], m->getMin()[1],  m->getMin()[2],  m->getMax()[0],  m->getMax()[1],  m->getMax()[2]);
	
}

void PathTracer::reset() {
	deleteImage(image); // Very important!
	image = newImage(renderCamera->resolution.x, renderCamera->resolution.y); 
}


Image* PathTracer::render() {
	Image* singlePassImage = newImage(image->width, image->height);

	launchKernel(singlePassImage->numPixels, singlePassImage->pixels, image->passCounter, *renderCamera); // Dereference not ideal.

	// TODO: Make a function for this (or a method---maybe Image can just be a class).
	for (int i = 0; i < image->numPixels; i++) {
		image->pixels[i] += singlePassImage->pixels[i];
	}
	image->passCounter++;

	deleteImage(singlePassImage);

	return image;
}
extern "C" void SetTexture(cudaArray *cuArray,int nCount);

void PathTracer::FirstSetTexture(unsigned int  pTexture,int &nCount){
		cudaGraphicsResource *cudaResource;
		cudaGraphicsGLRegisterImage(&cudaResource,pTexture, GL_TEXTURE_2D,cudaGraphicsRegisterFlagsNone);
		cudaError_t tmp=cudaGetLastError();
		cudaGraphicsMapResources(1,&(cudaResource), 0);
		if(tmp!=cudaSuccess)
			printf("%s\n",cudaGetErrorString(tmp));
		cudaArray *cuArray;
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray( &cuArray,cudaResource, 0, 0));
		SetTexture(cuArray,nCount);
		cudaGraphicsUnmapResources(1,&(cudaResource), 0);
}
void PathTracer::setUpScene() {

//	numSpheres = 14; // TODO: Move this!

}

void PathTracer::createDeviceData() {


}

void PathTracer::deleteDeviceData() {

}


