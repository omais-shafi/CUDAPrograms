#include <stdio.h>

__global__ void EmptyKernel() { }

int main() {

    const int N = 100000;

    float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i=0; i<N; i++) { 

        cudaEventRecord(start, 0);
        EmptyKernel<<<1,1>>>(); 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cumulative_time = cumulative_time + time;

    }

    printf("Kernel launch overhead time:  %3.5f ms \n", cumulative_time / N);
    return 0;
}
