#include "gl/glew.h"

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
#include "hdrloader.h"

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
	objLoader meshload("../bunny.obj", m);

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

void PathTracer::FirstSetTexture(unsigned int  pTexture,int nCount){
		cudaGraphicsResource *cudaResource;
		cudaGraphicsGLRegisterImage(&cudaResource,pTexture, GL_TEXTURE_2D,cudaGraphicsRegisterFlagsNone);
		cudaError_t tmp=cudaGetLastError();		
		if(tmp!=cudaSuccess)
			printf("%s\n",cudaGetErrorString(tmp));
		cudaGraphicsMapResources(1,&(cudaResource), 0);
		cudaArray *cuArray;
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray( &cuArray,cudaResource, 0, 0));
		SetTexture(cuArray,nCount);
		cudaGraphicsUnmapResources(1,&(cudaResource), 0);
}

void PathTracer::setEnvironmentMap(std::string filePath){
	vector<Image *>cubeMap;
	cubeMap.clear();

	const char *ext = &filePath[filePath.size()-4];
	assert(!strcmp(ext,".hdr"));
	HDRLoaderResult result;
	HDRLoader::load(filePath.c_str(),result);
	int tSize = result.width < result.height ? result.width : result.height;
	tSize /=3;
	for(int i = 0 ;i< 6;i++){
		cubeMap.push_back(newImage(tSize,tSize));
		memset(cubeMap[i]->pixels,0,sizeof(float3)*tSize*tSize);
	}
	if(result.width<result.height){
		loadSubImage(tSize ,tSize - 1,2 * tSize ,0  -1,result,cubeMap[2]);//posY
		loadSubImage(tSize - 1,tSize,0 - 1,2 * tSize,result,cubeMap[1]);//negX
		loadSubImage(2 * tSize - 1,tSize,tSize - 1,2 * tSize,result,cubeMap[5]);//negZ
		loadSubImage(3*tSize - 1,tSize ,2*tSize - 1,2*tSize ,result,cubeMap[0]);//posX
		loadSubImage(tSize , 3 * tSize - 1,2 * tSize , 2 * tSize - 1,result,cubeMap[3]);//negY;
		loadSubImage(tSize , 4 * tSize - 1,2 * tSize , 3 * tSize - 1,result,cubeMap[4]);//posZ
		// 	cubeMap[4]->SavePPM("posZ_hdr.ppm");
		//  	cubeMap[5]->SavePPM("negZ_hdr.ppm");
	}else
		assert(false);

////set texture to cuda
	GLuint texs[6];
	glGenTextures(6,texs);
	for(int i = 0;i<6;i++){
		glBindTexture(GL_TEXTURE_2D,texs[i]);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);    // (Modify This For The Type Of Filtering You Want)
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR); // (Modify This For The Type Of Filtering You Want)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, tSize, tSize, 0, GL_RGB, GL_FLOAT, cubeMap[i]->pixels);  // (Modify This If You Want Mipmaps)
		FirstSetTexture(texs[i],i+1);
	}
}
void PathTracer::loadSubImage(int fromX,int fromY,int toX,int toY,const HDRLoaderResult& data,Image *store){
	assert(fromX>=0&&fromX<data.width);
	assert(fromY>=0&&fromY<data.height);
	float3* image = store->pixels;
	if(!image)
		assert(false);
	int increasX = (toX - fromX) > 0?  1 : -1;
	int increasY = (toY - fromY) > 0?  1 : -1;

	for(int y = fromY ; y!=toY ; y+=increasY){
		for(int x = fromX; x!=toX ; x+=increasX){
			image[(abs(y - fromY)*store->width + abs(x - fromX))].x = data.cols[(y*data.width + x)*3];
			image[(abs(y - fromY)*store->width + abs(x - fromX))].y = data.cols[(y*data.width + x)*3 + 1];
			image[(abs(y - fromY)*store->width + abs(x - fromX))].z = data.cols[(y*data.width + x)*3 + 2];
		}
	}
}

void PathTracer::setUpScene() {
	setEnvironmentMap("../uffizi_cross.hdr");
//	numSpheres = 14; // TODO: Move this!

}

void PathTracer::createDeviceData() {


}

void PathTracer::deleteDeviceData() {

}


