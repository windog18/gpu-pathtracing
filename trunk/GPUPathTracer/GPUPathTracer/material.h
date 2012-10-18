#ifndef MATERIAL_H
#define MATERIAL_H

#include <cuda_runtime.h>
#include "color.h"
#include "medium.h"
class Material {
public:
	int ID;
	Color diffuseColor;
	Color emittedColor;
	Color specularColor;
	bool hasTransmission;
	Medium medium;
	__device__ __host__
	Material(){
		diffuseColor = make_float3(0.455, 0.43, 0.39);
		emittedColor = make_float3(0,0,0);
		specularColor = make_float3(0,0,0);
	    hasTransmission = false;
		medium.refractiveIndex = 0;
	    medium.absorptionAndScatteringProperties.absorptionCoefficient = make_float3(0,0,0);
	    medium.absorptionAndScatteringProperties.reducedScatteringCoefficient = 0;
	}
	__device__ __host__
	~Material(){
	}
};

// TODO: Figure out a way to do this without a macro! Ideally, figure out how to use classes in CUDA.
#define SET_DEFAULT_MATERIAL_PROPERTIES(material)															\
{																											\
	material.diffuseColor = make_float3(0,0,0);																\
	material.emittedColor = make_float3(0,0,0);																\
	material.specularColor = make_float3(0,0,0);															\
	material.hasTransmission = false;																		\
	SET_TO_AIR_MEDIUM(material.medium);																	\
}

/*
__host__ __device__
Material makeEmptyMaterial();
*/

#endif // MATERIAL_H