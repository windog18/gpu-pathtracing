#ifndef CUDA_SAFE_CALL_H
#define CUDA_SAFE_CALL_H

#include<cuda_runtime.h>
#include<stdio.h>

#define CUDA_SAFE_CALL( call) {											 \
cudaError err = call;                                                    \
if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
            __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
} }

inline void cutilSafeCall(cudaError err, const char *file = __FILE__,const int line = __LINE__){
	if(cudaSuccess!=err){
		fprintf(stderr,"%s(%i) : cudaSafeCall() RuntimeAPI error %d: %s.\n",file,line,(int)err,cudaGetErrorString(err));
		exit(-1);
	}

}

#endif // CUDA_SAFE_CALL_H