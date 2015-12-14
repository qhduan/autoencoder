#include "../inc/tool.h"


#include <iostream>

#include <cublas_v2.h>

namespace tensor {

cudaEvent_t start, stop;
float timeValue;

void StartTimer () {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );
}

float EndTimer () {
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  cudaEventElapsedTime( &timeValue, start, stop );
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  return timeValue;
}



}
