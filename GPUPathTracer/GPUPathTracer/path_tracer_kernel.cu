#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "driver_functions.h"
#include "sm_13_double_functions.h"


#include "cutil_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>

#include "basic_math.h"

#include "image.h"
#include "camera.h"
#include "sphere.h"
#include "poly.h"
#include "camera.h"
#include "fresnel.h"
#include "medium.h"
#include "absorption_and_scattering_properties.h"

#include "cuda_safe_call.h"

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include "geometry/KDTree.h"
#include "geometry/utils.h"
//using namespace cukd;
KdTree *kdtree;
namespace KD_TREE{
	extern "C" void active_ray_bunch_traverse(Knode *TreeNode, int *TriangleIndexUseForCuda, Triangle *treeTriangle, 
											 DevVector<Ray> &ray,DevVector<int> &activePixels,
											 DevVector<int> &hitID , DevVector<float> &lamda, DevVector<float> &U,DevVector<float> &V);
	extern "C" void active_ray_bunch_traverse_hits(Knode *TreeNode, int *TriangleIndexUseForCuda, Triangle *treeTriangle, 
												   DevVector<Ray> &ray,DevVector<int> &activePixels,
		                                           DevVector<int> &hitsID);
}
// Settings:
#define BLOCK_SIZE 256 // Number of threads in a block.
#define MAX_TRACE_DEPTH 30 // TODO: Put settings somewhere else and don't make them defines.
#define RAY_BIAS_DISTANCE 0.0002 // TODO: Put with other settings somewhere.
#define MIN_RAY_WEIGHT 0.00001 // Terminate rays below this weight.
#define HARD_CODED_GROUND_ELEVATION -0.8


// Numeric constants, copied from BasicMath:
#define PI                    3.1415926535897932384626422832795028841971
#define TWO_PI				  6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD     0.5773502691896257645091487805019574556476
#define E                     2.7182818284590452353602874713526624977572

texture<unsigned long, 2, cudaReadModeElementType> tex1;
texture<unsigned long, 2, cudaReadModeElementType> tex2;
texture<unsigned long, 2, cudaReadModeElementType> tex3;
texture<unsigned long, 2, cudaReadModeElementType> tex4;
texture<unsigned long, 2, cudaReadModeElementType> tex5;
texture<unsigned long, 2, cudaReadModeElementType> tex6;
texture<unsigned long, 2, cudaReadModeElementType> tex7;
texture<unsigned long, 2, cudaReadModeElementType> tex8;
texture<unsigned long, 2, cudaReadModeElementType> tex9;
texture<unsigned long, 2, cudaReadModeElementType> tex10;

extern "C" void SetTexture(cudaArray *cuArray,int nCount)
{
	tex1.addressMode[0] = cudaAddressModeWrap;
	tex1.addressMode[1] = cudaAddressModeWrap; 
	tex1.filterMode = cudaFilterModePoint; 
	tex1.normalized =true;
	tex2.addressMode[0] = cudaAddressModeWrap;
	tex2.addressMode[1] = cudaAddressModeWrap; 
	tex2.filterMode = cudaFilterModePoint; 
	tex2.normalized =true;
	tex3.addressMode[0] = cudaAddressModeWrap;
	tex3.addressMode[1] = cudaAddressModeWrap; 
	tex3.filterMode = cudaFilterModePoint; 
	tex3.normalized =true;
	tex4.addressMode[0] = cudaAddressModeWrap;
	tex4.addressMode[1] = cudaAddressModeWrap; 
	tex4.filterMode = cudaFilterModePoint; 
	tex4.normalized =true;
	tex5.addressMode[0] = cudaAddressModeWrap;
	tex5.addressMode[1] = cudaAddressModeWrap; 
	tex5.filterMode = cudaFilterModePoint; 
	tex5.normalized =true;
	tex6.addressMode[0] = cudaAddressModeWrap;
	tex6.addressMode[1] = cudaAddressModeWrap; 
	tex6.filterMode = cudaFilterModePoint; 
	tex6.normalized =true;
	tex7.addressMode[0] = cudaAddressModeWrap;
	tex7.addressMode[1] = cudaAddressModeWrap; 
	tex7.filterMode = cudaFilterModePoint; 
	tex7.normalized =true;
	tex8.addressMode[0] = cudaAddressModeWrap;
	tex8.addressMode[1] = cudaAddressModeWrap; 
	tex8.filterMode = cudaFilterModePoint; 
	tex8.normalized =true;
	tex9.addressMode[0] = cudaAddressModeWrap;
	tex9.addressMode[1] = cudaAddressModeWrap; 
	tex9.filterMode = cudaFilterModePoint; 
	tex9.normalized =true;
	tex10.addressMode[0] = cudaAddressModeWrap;
	tex10.addressMode[1] = cudaAddressModeWrap; 
	tex10.filterMode = cudaFilterModePoint; 
	tex10.normalized =true;
	cudaChannelFormatDesc chDesc=cudaCreateChannelDesc<unsigned long>();
/*	unsigned  int w=40,h=40,size=w*h*sizeof(unsigned long);
	cudaArray *tcuArray;
	unsigned long *temp=new unsigned long[w*h];
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			 BYTE *pPixel=( BYTE *)(&temp[i*w+j]);
			 BYTE c=(((i&0x2)==0)^((j&0x2)==0))*255;
			 pPixel[0]=c;
			 pPixel[1]=c;
			 pPixel[2]=c;
			 pPixel[3]=(BYTE)255;

		}
	cutilSafeCall(cudaMallocArray(&tcuArray,&chDesc,w,h));
	cutilSafeCall(cudaMemcpyToArray(tcuArray,0,0,temp,size,cudaMemcpyHostToDevice));*/
//cudaMemcpyToArray(tcuArray,0,0,cuArray,w*h*sizeof(unsigned long),cudaMemcpyDeviceToDevice);
//	cudaGetChannelDesc(&p,cuArray);
	if(nCount==1)
	cutilSafeCall(cudaBindTextureToArray(tex1,cuArray,chDesc));
	else if(nCount==2)
	cutilSafeCall(cudaBindTextureToArray(tex2,cuArray,chDesc));
	else if(nCount==3)
	cutilSafeCall(cudaBindTextureToArray(tex3,cuArray,chDesc));
	else if(nCount==4)
	cutilSafeCall(cudaBindTextureToArray(tex4,cuArray,chDesc));
	else if(nCount==5)
	cutilSafeCall(cudaBindTextureToArray(tex5,cuArray,chDesc));
	else if(nCount==6)
	cutilSafeCall(cudaBindTextureToArray(tex6,cuArray,chDesc));
	else if(nCount==7)
	cutilSafeCall(cudaBindTextureToArray(tex7,cuArray,chDesc));
	else if(nCount == 8)
	cutilSafeCall(cudaBindTextureToArray(tex8,cuArray,chDesc));
	else if(nCount == 9)
	cutilSafeCall(cudaBindTextureToArray(tex9,cuArray,chDesc));
	else
    cutilSafeCall(cudaBindTextureToArray(tex10,cuArray,chDesc));

}



__host__ __device__
unsigned int randhash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__global__ void initializeThings(int numPixels, int* activePixels, AbsorptionAndScatteringProperties* absorptionAndScattering, Color* notAbsorbedColors, Color* accumulatedColors) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	if (validIndex) {

		activePixels[pixelIndex] = pixelIndex;
		SET_TO_AIR_ABSORPTION_AND_SCATTERING_PROPERTIES(absorptionAndScattering[pixelIndex]);
		notAbsorbedColors[pixelIndex] = make_float3(1,1,1);
		accumulatedColors[pixelIndex] = make_float3(0,0,0);

	}

}

__global__ void raycastFromCameraKernel(Camera renderCamera, int numPixels,Ray *rays, unsigned long seed) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	if (validIndex) {

		// get x and y coordinates of pixel
		int y = int(pixelIndex/renderCamera.resolution.y);
		int x = pixelIndex - (y*renderCamera.resolution.y);
	
		// generate random jitter offsets for supersampled antialiasing
		thrust::default_random_engine rng( randhash(seed) * randhash(pixelIndex) * randhash(seed) );
		thrust::uniform_real_distribution<float> uniformDistribution(0.0, 1.0); // Changed to 0.0 and 1.0 so I could reuse it for aperture sampling below.
		float jitterValueX = uniformDistribution(rng) - 0.5; 
		float jitterValueY = uniformDistribution(rng) - 0.5; 

		// compute important values
		renderCamera.view = normalize(renderCamera.view); // view is already supposed to be normalized, but normalize it explicitly just in case.
		float3 horizontalAxis = cross(renderCamera.view, renderCamera.up);
		horizontalAxis = normalize(horizontalAxis); // Important!
		float3 verticalAxis = cross(horizontalAxis, renderCamera.view); 
		verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.
		
		// compute point on image plane
		float3 middle = renderCamera.position + renderCamera.view;
		float3 horizontal = horizontalAxis * tan(renderCamera.fov.x * 0.5 * (PI/180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		float3 vertical = verticalAxis * tan(-renderCamera.fov.y * 0.5 * (PI/180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		
		float sx = (jitterValueX+x)/(renderCamera.resolution.x-1);
		float sy = (jitterValueY+y)/(renderCamera.resolution.y-1);

		float3 pointOnPlaneOneUnitAwayFromEye = middle + ( ((2*sx)-1) * horizontal) + ( ((2*sy)-1) * vertical);

		// move and resize image plane based on focalDistance
		// could also incorporate this into the original computations of the point
		float3 pointOnImagePlane = renderCamera.position + ( (pointOnPlaneOneUnitAwayFromEye - renderCamera.position) * renderCamera.focalDistance ); // Important for depth of field!

		// now compute the point on the aperture (or lens)
		float3 aperturePoint;
		if (renderCamera.apertureRadius > 0.00001) { // The small number is an epsilon value.
			// generate random numbers for sampling a point on the aperture
			float random1 = uniformDistribution(rng);
			float random2 = uniformDistribution(rng);

			// sample a point on the circular aperture
			float angle = TWO_PI * random1;
			float distance = renderCamera.apertureRadius * sqrt(random2);

			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;

			aperturePoint = renderCamera.position + (apertureX * horizontalAxis) + (apertureY * verticalAxis);
		} else {
			aperturePoint = renderCamera.position;
		}
		aperturePoint = renderCamera.position;
		float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;

		rays[pixelIndex].m_unitDir  = normalize(pointOnPlaneOneUnitAwayFromEye - aperturePoint);
		rays[pixelIndex].m_startPos = aperturePoint;

		//accumulatedColors[pixelIndex] = rays[pixelIndex].m_unitDir;	//test code, should output green/yellow/black/red if correct
	}
}

__host__ __device__
float3 positionAlongRay(const Ray & ray, float t) {
	Vec3 pos = ray.m_startPos + t * ray.m_unitDir;
	return pos.toFloat3();
}

__host__ __device__
float findGroundPlaneIntersection(float elevation, const Ray & ray, float3 & normal) {

	if (ray.m_unitDir.y != 0) {

		double t = (elevation - ray.m_startPos.y) / ray.m_unitDir.y;

		if (ray.m_unitDir.y < 0) { // Top of plane.
			normal = make_float3(0, 1, 0);
		} else { // Bottom of plane.
			normal = make_float3(0, 1, 0);//make_float3(0, -1, 0); // Make the normal negative for opaque appearance. Positive normal lets you see diffusely through the ground which looks really cool and gives you a better idea of where you are!
		}
		
		return t;

	}

	
	return -1; // No intersection.
}

//dev_RayTriIntersect: triangle intersection test
//a/b/c are vertices of triangle
//o is ray origin, d is ray direction
//out_lambda is intersection parameter lambda, out_bary1/2 are barycentric hit coordinates
//adapted from MNRT code by Mathias Neumann
inline __device__ bool FindTriangleIntersect(const float3 a, const float3 b, const float3 c, 
										   const float3 o, const float3 d,
										   float& out_lambda, float& out_bary1, float& out_bary2)
{
	float3 edge1 = b - a;
	float3 edge2 = c - a;

	float3 pvec = cross(d, edge2);
	float det = dot(edge1, pvec);
	if(det == 0.f)
		return false;
	float inv_det = 1.0f / det;

	float3 tvec = o - a;
	out_bary1 = dot(tvec, pvec) * inv_det;

	float3 qvec = cross(tvec, edge1);
	out_bary2 = dot(d, qvec) * inv_det;
	out_lambda = dot(edge2, qvec) * inv_det;

	bool hit = (out_bary1 >= 0.0f && out_bary2 >= 0.0f && (out_bary1 + out_bary2) <= 1.0f);
	return hit;
}



__host__ __device__
float3 randomCosineWeightedDirectionInHemisphere(const float3 & normal, float xi1, float xi2) {

    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;

	// Find any two perpendicular m_unitDirs:
	// Either all of the components of the normal are equal to the square root of one third, or at least one of the components of the normal is less than the square root of 1/3.
	float3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) { 
		directionNotNormal = make_float3(1, 0, 0);
	} else if (abs(normal.y) < SQRT_OF_ONE_THIRD) { 
		directionNotNormal = make_float3(0, 1, 0);
	} else {
		directionNotNormal = make_float3(0, 0, 1);
	}
	float3 perpendicular1 = normalize( cross(normal, directionNotNormal) );
	float3 perpendicular2 =            cross(normal, perpendicular1); // Normalized by default.
  
    return ( up * normal ) + ( cos(around) * over * perpendicular1 ) + ( sin(around) * over * perpendicular2 );

}

__host__ __device__
float3 randomDirectionInSphere(float xi1, float xi2) {

    float up = xi1 * 2 - 1; // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;

    return make_float3( up, cos(around) * over, sin(around) * over );

}

////rotate with arbitrarily direction
__host__ __device__ 
float3 matR(float3 curDir,float theta,float x,float y,float z)
{
	float c=cos(theta),s=sin(theta),onec=1-c;
	float len=sqrt(x*x+y*y+z*z);
	if(len<=0)  return make_float3(0,0,0);
	float ux=x/len,uy=y/len,uz=z/len;

	int i,j;
	float3 result;
	result.x = curDir.x * (ux*ux*onec+c) + curDir.y * (ux*uy*onec-uz*s) + curDir.z * (ux*uz*onec+uy*s);
	result.y = curDir.x * (uy*ux*onec+uz*s) + curDir.y * (uy*uy*onec+c) + curDir.z * (uy*uz*onec-ux*s);
	result.z = curDir.x * (uz*ux*onec-uy*s) + curDir.y * (uz*uy*onec+ux*s) + curDir.z * (uz*uz*onec+c);
	return result;
}


__host__ __device__ 
void CoordinateSystem(const float3 &v1, float3 *v2, float3 *v3) {
    if (fabsf(v1.x) > fabsf(v1.y)) {
        float invLen = 1.f / sqrtf(v1.x*v1.x + v1.z*v1.z);
        *v2 = make_float3(-v1.z * invLen, 0.f, v1.x * invLen);
    }
    else {
        float invLen = 1.f / sqrtf(v1.y*v1.y + v1.z*v1.z);
        *v2 = make_float3(0.f, v1.z * invLen, -v1.y * invLen);
    }
    *v3 = cross(v1, *v2);
}

__host__ __device__
float3 randomDircetionHG(float g,float3 curDir,float rand1,float rand2){
	float s = rand1 * 2 - 1;
	float u = (1 - g * g)/(1 + g * s) ;
	u *=u;
	u = 1 + g * g - u;
	u = u/(2 * g);
	float theta = acos(u);
	float phi   = rand2 * PI * 2;
	float3 v2,v3;
	CoordinateSystem(curDir,&v2,&v3);
	float3 result;
  	result = matR(curDir,theta,v2.x,v2.y,v2.z);
	result = matR(result,phi,curDir.x,curDir.y,curDir.z);
	return result;
}

__host__ __device__
float3 computeReflectionDirection(const float3 & normal, const float3 & incident) {
	return 2.0 * dot(normal, incident) * normal - incident;
}

__host__ __device__
float3 computeTransmissionDirection(const float3 & normal, const float3 & incident, float refractiveIndexIncident, float refractiveIndexTransmitted) {
	// Snell's Law:
	// Copied from Photorealizer.

	float cosTheta1 = dot(normal, incident);

	float n1_n2 =  refractiveIndexIncident /  refractiveIndexTransmitted;

	float radicand = 1 - pow(n1_n2, 2) * (1 - pow(cosTheta1, 2));
	if (radicand < 0) return make_float3(0, 0, 0); // Return value???????????????????????????????????????
	float cosTheta2 = sqrt(radicand);

	if (cosTheta1 > 0) { // normal and incident are on same side of the surface.
		return n1_n2 * (-1 * incident) + ( n1_n2 * cosTheta1 - cosTheta2 ) * normal;
	} else { // normal and incident are on opposite sides of the surface.
		return n1_n2 * (-1 * incident) + ( n1_n2 * cosTheta1 + cosTheta2 ) * normal;
	}

}

__host__ __device__
Fresnel computeFresnel(const float3 & normal, const float3 & incident, float refractiveIndexIncident, float refractiveIndexTransmitted, const float3 & reflectionDirection, const float3 & transmissionDirection) {
	Fresnel fresnel;


	
	// First, check for total internal reflection:
	if ( length(transmissionDirection) <= 0.12345 || dot(normal, transmissionDirection) > 0 ) { // The length == 0 thing is how we're handling TIR right now.
		// Total internal reflection!
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
		return fresnel;
	}



	// Real Fresnel equations:
	// Copied from Photorealizer.
	float cosThetaIncident = dot(normal, incident);
	float cosThetaTransmitted = dot(-1 * normal, transmissionDirection);
	float reflectionCoefficientSPolarized = pow(   (refractiveIndexIncident * cosThetaIncident - refractiveIndexTransmitted * cosThetaTransmitted)   /   (refractiveIndexIncident * cosThetaIncident + refractiveIndexTransmitted * cosThetaTransmitted)   , 2);
    float reflectionCoefficientPPolarized = pow(   (refractiveIndexIncident * cosThetaTransmitted - refractiveIndexTransmitted * cosThetaIncident)   /   (refractiveIndexIncident * cosThetaTransmitted + refractiveIndexTransmitted * cosThetaIncident)   , 2);
	float reflectionCoefficientUnpolarized = (reflectionCoefficientSPolarized + reflectionCoefficientPPolarized) / 2.0; // Equal mix.
	//
	fresnel.reflectionCoefficient = reflectionCoefficientUnpolarized;
	fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
	return fresnel;
	
	/*
	// Shlick's approximation including expression for R0 and modification for transmission found at http://www.bramz.net/data/writings/reflection_transmission.pdf
	// TODO: IMPLEMENT ACTUAL FRESNEL EQUATIONS!
	float R0 = pow( (refractiveIndexIncident - refractiveIndexTransmitted) / (refractiveIndexIncident + refractiveIndexTransmitted), 2 ); // For Schlick's approximation.
	float cosTheta;
	if (refractiveIndexIncident <= refractiveIndexTransmitted) {
		cosTheta = dot(normal, incident);
	} else {
		cosTheta = dot(-1 * normal, transmissionDirection); // ???
	}
	fresnel.reflectionCoefficient = R0 + (1.0 - R0) * pow(1.0 - cosTheta, 5); // Costly pow function might make this slower than actual Fresnel equations. TODO: USE ACTUAL FRESNEL EQUATIONS!
	fresnel.transmissionCoefficient = 1.0 - fresnel.reflectionCoefficient;
	return fresnel;
	*/

}

__host__ __device__
Color computeBackgroundColor(const float3 & direction) {
/*	float position = (dot(direction, normalize(make_float3(-0.5, 0.5, -1.0))) + 1) / 2;
	Color firstColor = make_float3(0.15, 0.3, 0.5); // Bluish.
	Color secondColor = make_float3(1.0, 1.0, 1.0); // White.
	Color interpolatedColor = (1 - position) * firstColor + position * secondColor;
	float radianceMultiplier = 1.0;

	return interpolatedColor * radianceMultiplier;*/
	if(direction.z > 0)
		return make_float3(1,1,1);
}

__host__ __device__
Color computeTransmission(Color absorptionCoefficient, float distance) {
	Color transmitted;
	transmitted.x = pow((float)E, (float)(-1 * absorptionCoefficient.x * distance));
	transmitted.y = pow((float)E, (float)(-1 * absorptionCoefficient.y * distance));
	transmitted.z = pow((float)E, (float)(-1 * absorptionCoefficient.z * distance));
	return transmitted;
}

__global__ void traceRayKernel(Triangle *m_triangles,int numActivePixels, int* activePixels, Ray *rays,
	                           int *isectID,float *alpha,int rayDepth, AbsorptionAndScatteringProperties* absorptionAndScattering, 
							   float3* notAbsorbedColors, float3* accumulatedColors, unsigned long seedOrPass) {

//__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	// Duplicate code:
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int activePixelIndex = BLOCK_SIZE * bx + tx;
	
	if (activePixelIndex >= numActivePixels) return;
	// TODO: Restructure stuff! It's a mess. Use classes!

	int pixelIndex  = activePixels[activePixelIndex];
	Ray currentRay  = rays[pixelIndex];

	int intersectID = isectID[pixelIndex];

	thrust::default_random_engine rng( randhash(seedOrPass) * randhash(pixelIndex) * randhash(rayDepth) );	
	thrust::uniform_real_distribution<float> uniformDistribution(0,1);

	float bestT = 123456789; // floatInfinity(); // std::numeric_limits<float>::infinity();
	float3 bestNormal;// = make_float3(0,0,0);
	bool bestIsGroundPlane = false;
	bool bestIsSphere = false;
	bool bestIsPoly = false;
	int bestSphereIndex = -1;
	int bestPolyIndex = -1;

	// Reusables:
	float t;
	float3 normal;

	// Check for ground plane intersection:
	t = findGroundPlaneIntersection(HARD_CODED_GROUND_ELEVATION, currentRay, normal); 
	if (t > 0) { // No "<" conditional only because this is being tested before anything else.
		bestT = t;
		bestNormal = normal;

		bestIsGroundPlane = true;
		bestIsSphere = false;
		bestIsPoly = false;
	}

		float pt;
		float ob1;
		float ob2;
		bool tr = false;
	/*	int intersectID2 = -1;
		for(int i = 0;i < 12;i++){
		   tr = FindTriangleIntersect(m_triangles[i].v1.toFloat3(),m_triangles[i].v2.toFloat3(),m_triangles[i].v3.toFloat3(),
			                      currentRay.m_startPos.toFloat3(),currentRay.m_unitDir.toFloat3(),pt,ob1,ob2);
		   if(tr){
			   if(pt>0 && pt < bestT){
				   bestT = t;
				   intersectID2 = i;
			   }
		   }
		}*/
		if(intersectID >=0){
		    pt = alpha[pixelIndex];
			//accumulatedColors[pixelIndex] = make_float3(1,0,0);
			tr = true;
		}
		//return ;
		if(tr){
			t = pt;
			if (t > 0 && t < bestT) {
				bestT = t;
				Vec3 temp = (m_triangles[intersectID].n1 + m_triangles[intersectID].n2 + m_triangles[intersectID].n3)/3;
				bestNormal = temp.toFloat3();

				bestIsGroundPlane = false;
				bestIsSphere = false;
				bestIsPoly = true;

				bestPolyIndex = intersectID;
			}
		}
						
	// ABSORPTION AND SCATTERING:
	{ // BEGIN SCOPE.
		AbsorptionAndScatteringProperties currentAbsorptionAndScattering = absorptionAndScattering[pixelIndex];

		#define ZERO_ABSORPTION_EPSILON 0.00001
		//if(0){
		if ( currentAbsorptionAndScattering.reducedScatteringCoefficient > 0 || dot(currentAbsorptionAndScattering.absorptionCoefficient, currentAbsorptionAndScattering.absorptionCoefficient) > ZERO_ABSORPTION_EPSILON ) { // The dot product with itself is equivalent to the squre of the length.
			
		    float randomFloatForScatteringDistance = uniformDistribution(rng);
			float scatteringDistance = -log(randomFloatForScatteringDistance) / absorptionAndScattering[pixelIndex].reducedScatteringCoefficient;
			if (scatteringDistance < bestT) {
				// Both absorption and scattering.

				// Scatter the ray:
				Ray nextRay;
				nextRay.m_startPos = positionAlongRay(currentRay, scatteringDistance);
				float random1 = uniformDistribution(rng);
				float random2 = uniformDistribution(rng);
				nextRay.m_unitDir = //randomDircetionHG(currentAbsorptionAndScattering.g,currentRay.m_unitDir.toFloat3(),random1,random2);
				                  randomDirectionInSphere(random1, random2); // Isoptropic scattering!
				rays[pixelIndex].m_startPos   = nextRay.m_startPos; // Only assigning to global memory ray once, for better performance.
				rays[pixelIndex].m_unitDir    = nextRay.m_unitDir;
				// Compute how much light was absorbed along the ray before it was scattered:
				notAbsorbedColors[pixelIndex] *= computeTransmission(currentAbsorptionAndScattering.absorptionCoefficient, scatteringDistance);

				// DUPLICATE CODE:
				// To assist Thrust stream compaction, set this activePixel to -1 if the ray weight is now zero:
				if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // TODO: Faster: dot product of a vector with itself is the same as its length squared.
					activePixels[activePixelIndex] = -1;
				}

				// That's it for this iteration!
				return; // IMPORTANT!
			} else {
				// Just absorption.
				notAbsorbedColors[pixelIndex] *= computeTransmission(currentAbsorptionAndScattering.absorptionCoefficient, bestT);

				// Now proceed to compute interaction with intersected object and whatnot!
			}
		}
	} // END SCOPE.


	if (bestIsGroundPlane || bestIsSphere || bestIsPoly) {

		
		
		Material bestMaterial;
		SET_DEFAULT_MATERIAL_PROPERTIES(bestMaterial);
		if (bestIsGroundPlane) {
			Material hardCodedGroundMaterial;
			SET_DEFAULT_MATERIAL_PROPERTIES(hardCodedGroundMaterial);
			hardCodedGroundMaterial.diffuseColor = make_float3(0.455, 0.43, 0.39);
			hardCodedGroundMaterial.emittedColor = make_float3(0,0,0);
			hardCodedGroundMaterial.specularColor = make_float3(0,0,0);
			hardCodedGroundMaterial.hasTransmission = false;
			bestMaterial = hardCodedGroundMaterial;
		} else if (bestIsSphere) {
		//	bestMaterial = spheres[bestSphereIndex].material;
		} else if(bestIsPoly){
			Material marble;
			SET_DEFAULT_MATERIAL_PROPERTIES(marble);
			marble.specularColor = make_float3(0.83, 0.79, 0.75);
			marble.hasTransmission = true;
			marble.diffuseColor = make_float3(0.83, 0.79, 0.75);
			marble.medium.refractiveIndex = 1.5;
			marble.medium.absorptionAndScatteringProperties.absorptionCoefficient = make_float3(0.0021 ,0.0041 ,0.0071)*5;
			marble.medium.absorptionAndScatteringProperties.reducedScatteringCoefficient = 3.00 *5;
			marble.medium.absorptionAndScatteringProperties.g = 0.01;
			bestMaterial = marble;
		//	bestMaterial = polys[bestPolyIndex].material;
		}




		// TODO: Reduce duplicate code and memory usage here and in the functions called here.
		// TODO: Finish implementing the functions called here.
		// TODO: Clean all of this up!
		float3 incident = -currentRay.m_unitDir.toFloat3();

		Medium incidentMedium;
		SET_TO_AIR_MEDIUM(incidentMedium);
		Medium transmittedMedium = bestMaterial.medium;

		bool backFace = ( dot(bestNormal, incident) < 0 );

		if (backFace) {
			// Flip the normal:
			bestNormal *= -1;
			// Swap the IORs:
			// TODO: Use the BasicMath swap function if possible on the device.
			Medium tempMedium = incidentMedium;
			incidentMedium = transmittedMedium;
			transmittedMedium = tempMedium;
		}

		float3 reflectionDirection = computeReflectionDirection(bestNormal, incident);
		float3 transmissionDirection = computeTransmissionDirection(bestNormal, incident, incidentMedium.refractiveIndex, transmittedMedium.refractiveIndex);

		float3 bestIntersectionPoint = positionAlongRay(currentRay, bestT);

		float3 biasVector = ( RAY_BIAS_DISTANCE * bestNormal ); // TODO: Bias ray in the other direction if the new ray is transmitted!!!

		bool doSpecular = ( bestMaterial.medium.refractiveIndex > 1.0 ); // TODO: Move?
		float rouletteRandomFloat = uniformDistribution(rng);
		// TODO: Fix long conditional, and maybe lots of temporary variables.
		// TODO: Optimize total internal reflection case (no random number necessary in that case).
		bool reflectFromSurface = ( doSpecular && rouletteRandomFloat < computeFresnel(bestNormal, incident, incidentMedium.refractiveIndex, transmittedMedium.refractiveIndex, reflectionDirection, transmissionDirection).reflectionCoefficient );
		if (reflectFromSurface) {
			// Ray reflected from the surface. Trace a ray in the reflection direction.

			// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
			notAbsorbedColors[pixelIndex] *= bestMaterial.specularColor;

			Ray nextRay;
			nextRay.m_startPos = bestIntersectionPoint + biasVector;
			nextRay.m_unitDir = reflectionDirection;
			rays[pixelIndex].m_startPos    = nextRay.m_startPos; // Only assigning to global memory ray once, for better performance.
			rays[pixelIndex].m_unitDir     = nextRay.m_unitDir;

		} else if (bestMaterial.hasTransmission) {
			// Ray transmitted and refracted.

			// The ray has passed into a new medium!
			absorptionAndScattering[pixelIndex] = transmittedMedium.absorptionAndScatteringProperties;

			Ray nextRay;
			nextRay.m_startPos = bestIntersectionPoint - biasVector; // Bias ray in the other direction because it's transmitted!!!
			nextRay.m_unitDir = transmissionDirection;
			rays[pixelIndex].m_startPos    = nextRay.m_startPos; // Only assigning to global memory ray once, for better performance.
			rays[pixelIndex].m_unitDir     = nextRay.m_unitDir;
		} else {
			// Ray did not reflect from the surface, so consider emission and take a diffuse sample.

			// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
			accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * bestMaterial.emittedColor;
			notAbsorbedColors[pixelIndex] *= bestMaterial.diffuseColor;

			// Choose a new ray direction:
			float randomFloat1 = uniformDistribution(rng); 
			float randomFloat2 = uniformDistribution(rng); 
			Ray nextRay;
			nextRay.m_startPos = bestIntersectionPoint + biasVector;
			nextRay.m_unitDir  = randomCosineWeightedDirectionInHemisphere(bestNormal, randomFloat1, randomFloat2);
			rays[pixelIndex].m_startPos    = nextRay.m_startPos; // Only assigning to global memory ray once, for better performance.
			rays[pixelIndex].m_unitDir     = nextRay.m_unitDir;
		//	printf("%d\n",intersectID);
		}


		// DUPLICATE CODE:
		// To assist Thrust stream compaction, set this activePixel to -1 if the ray weight is now zero:
		if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // TODO: Faster: dot product of a vector with itself is the same as its length squared.
			activePixels[activePixelIndex] = -1;
		}
	} else {
		// Ray didn't hit an object, so sample the background and terminate the ray.
		accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * computeBackgroundColor(currentRay.m_unitDir.toFloat3());
		//notAbsorbedColors[pixelIndex] = make_float3(0,0,0); // The ray now has zero weight. // TODO: Remove this? This isn't even necessary because we know the ray will be terminated anyway.

		activePixels[activePixelIndex] = -1; // To assist Thrust stream compaction, set this activePixel to -1 because the ray weight is now zero.
	}
	// MOVED:
	//// To assist Thrust stream compaction, set this activePixel to -1 if the ray weight is now zero:
	//if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // Faster: dot product of a vector with itself is the same as its length squared.
	//	activePixels[activePixelIndex] = -1;
	//}


}



// http://docs.thrust.googlecode.com/hg/group__counting.html
// http://docs.thrust.googlecode.com/hg/group__stream__compaction.html
struct isNegative
{
	__host__ __device__ 
	bool operator()(const int & x) 
	{
		return x < 0;
	}
};


extern "C"
void constructKDTree(std::vector<Triangle> &tris, float minx, float miny, float minz, float maxx, float maxy, float maxz){

	std::cout << "Building GPU KD-Tree..." << std::endl;
	kdtree = new KdTree(tris.size() * 3,tris.size());
	for(int i = 0;i < tris.size();i++){
		kdtree->AddTriangleToScene(&tris[i]);
	}
	kdtree->UseSAHToCreateTree();
	kdtree->transformTreeToGPU();
	std::cout << "Built GPU KD-Tree!" << std::endl;
	//kdtree->print_preorder();
}



extern "C"
void launchKernel(int numPixels, Color* pixels, int counter, Camera renderCamera) {
	


	// Configure grid and block sizes:
	int threadsPerBlock = BLOCK_SIZE;

	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int fullBlocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;





	// Declare and allocate active pixels, color arrays, and rays:
	AbsorptionAndScatteringProperties* absorptionAndScattering = NULL;
	Color* notAbsorbedColors = NULL;
	Color* accumulatedColors = NULL;
	CUDA_SAFE_CALL( cudaMalloc((void**)&absorptionAndScattering, numPixels * sizeof(AbsorptionAndScatteringProperties)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&notAbsorbedColors, numPixels * sizeof(Color)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&accumulatedColors, numPixels * sizeof(Color)) );
	DevVector<int> hits;
    DevVector<int> costs;
    DevVector<float> alpha, x1, x2;
	DevVector<Ray> rays;
	DevVector<int> activePixels;
	hits.resize(numPixels);
    costs.resize(numPixels);
    alpha.resize(numPixels);
    x1.resize(numPixels);
    x2.resize(numPixels);
	rays.resize(numPixels);
	activePixels.resize(numPixels);
	initializeThings<<<fullBlocksPerGrid, threadsPerBlock>>>(activePixels.size() , activePixels.pointer(), absorptionAndScattering, notAbsorbedColors, accumulatedColors);

	int numActivePixels = numPixels;
	
	// Run this every pass so we can do anti-aliasing using jittering.
	// If we don't want to re-compute the camera rays, we'll need a separate array for secondary rays.
	
	raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCamera, activePixels.size(), rays.pointer(), counter);





	for (int rayDepth = 0; rayDepth < MAX_TRACE_DEPTH; rayDepth++) {

		// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
		int newBlocksPerGrid = (numActivePixels + threadsPerBlock - 1) / threadsPerBlock; // Duplicate code.
		KD_TREE::active_ray_bunch_traverse(kdtree->TreeNode_device,kdtree->TriangleIndexUseForCuda_device,
			                               kdtree->treeTriangle_device,rays,activePixels,hits,alpha,x1,x2);
		 traceRayKernel<<<newBlocksPerGrid, threadsPerBlock>>>(kdtree->treeTriangle_device, numActivePixels, activePixels.pointer(), 
			 rays.pointer(), hits.pointer(), alpha.pointer(), rayDepth, absorptionAndScattering, notAbsorbedColors, accumulatedColors, counter);
		 cudaThreadSynchronize();


		// Use Thrust stream compaction to compress activePixels:
		// http://docs.thrust.googlecode.com/hg/group__stream__compaction.html#ga5fa8f86717696de88ab484410b43829b
		/*thrust::remove_if(activePixels.begin(), activePixels.end(), isNegative());
		thrust::device_ptr<int> devicePointer(activePixels.pointer());
		thrust::device_ptr<int> newEnd = thrust::remove_if(devicePointer, devicePointer + numActivePixels, isNegative());
		numActivePixels = activePixels.size();*/
		thrust::device_ptr<int> devicePointer(activePixels.pointer());
		thrust::device_ptr<int> newEnd = thrust::remove_if(devicePointer, devicePointer + numActivePixels, isNegative());
		numActivePixels = newEnd.get() - activePixels.pointer();
		activePixels.resize(numActivePixels);
		std::cout << numActivePixels << std::endl;
		if(numActivePixels == 0)
		   break;
	}

	// Copy the accumulated colors from the device into the host image:
	CUDA_SAFE_CALL( cudaMemcpy( pixels, accumulatedColors, numPixels * sizeof(Color), cudaMemcpyDeviceToHost) );



	// Clean up:
	// TODO: Save these things for the next iteration!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	CUDA_SAFE_CALL( cudaFree( absorptionAndScattering ) );
	CUDA_SAFE_CALL( cudaFree( notAbsorbedColors ) );
	CUDA_SAFE_CALL( cudaFree( accumulatedColors ) );


}
