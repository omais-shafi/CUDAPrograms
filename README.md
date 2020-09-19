# CUDAPrograms
It contains sample cuda programs to measure the execution time of the kernels and observe the difference between the time reported by nvprof and the cudaEventTimer

launchoverhead.cu  -----> It launches the empty kernel and we can obtain the overhead of the empty kernel.

matrixmul.cu -----> It does matrix multiplication and contains one kernel for that. It also does data transfer from host to device and vice-versa.

matrixkernel.cu -----> Does multiple kernel invocations of the same kernel and reports the overall time from the memory transfer to the kernel invocations

matrixkernel_data.cu ----> Does multiple kernel invocations and reports the individual time for Host to Device transfer, kernel time and the Device to Host Transfer.

Note: We can do multiple changes to the cuda code and observe the difference of the time reported by both nvprof and cudaEventTimer


#Requirements to run these programs

You need to ensure
1. Cuda is installed(Tested on cuda-10.0)
2. nvprof is installed(If cuda is installed, you can found nvprof in the bin directory of cuda)


#How to run

1. First compile the cuda program using the nvcc compiler(Present by default in cuda)

   nvcc <testprogram.cu> -o executable
  
2. Execute the executable using nvprof
   
   nvprof executable
  

