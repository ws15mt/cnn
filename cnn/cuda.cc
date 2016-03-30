#include <iostream>

#include "cnn/cnn.h"
#include "cnn/cuda.h"

#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cudart_static.lib")
/// need to include library C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\x64 that has cublas.lib to project
#pragma comment(lib, "cudnn.lib")

using namespace std;

namespace cnn {

cublasHandle_t cublas_handle;
cudnnHandle_t cudnn_handle;
cudnnDataType_t cudnnDataType;

void Initialize_CUDNN()
{
    cudnn_handle = nullptr;
    if (sizeof(cnn::real) == sizeof(float) )
        cudnnDataType = CUDNN_DATA_FLOAT; 
    else if (sizeof(cnn::real) == sizeof(double))
        cudnnDataType = CUDNN_DATA_DOUBLE;
    else
        throw std::exception("not supported data type");

    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
}

void Initialize_Consts_And_Store_In_GPU()
{
    kSCALAR_ONE_OVER_INT.resize(MEM_PRE_ALLOCATED_CONSTS_NUMBERS);
    for (int i = 0; i < MEM_PRE_ALLOCATED_CONSTS_NUMBERS; i++)
    {
        cnn::real flt = 1./(i+2);
        cnn::real *flt_val; 
        CUDA_CHECK(cudaMalloc(&flt_val, sizeof(cnn::real)));
        CUDA_CHECK(cudaMemcpyAsync(flt_val, &flt, sizeof(cnn::real), cudaMemcpyHostToDevice));
        kSCALAR_ONE_OVER_INT[i] = flt_val;
    }
}

void Free_GPU()
{
#ifdef HAVE_CUDA
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
#endif
}

void Initialize_GPU(int& argc, char**& argv) {
  int nDevices;
  CUDA_CHECK(cudaGetDeviceCount(&nDevices));
  if (nDevices < 1) {
    cerr << "[cnn] No GPUs found, recompile without DENABLE_CUDA=1\n";
    throw std::runtime_error("No GPUs found but CNN compiled with CUDA support.");
  }
  size_t free_bytes, total_bytes, max_free = 0;
  int selected = 0;
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    cerr << "[cnn] Device Number: " << i << endl;
    cerr << "[cnn]   Device name: " << prop.name << endl;
    cerr << "[cnn]   Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
    cerr << "[cnn]   Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
    cerr << "[cnn]   Peak Memory Bandwidth (GB/s): " << (2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6) << endl << endl;
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaMemGetInfo( &free_bytes, &total_bytes ));
    CUDA_CHECK(cudaDeviceReset());
    cerr << "[cnn]   Memory Free (MB): " << (int)free_bytes/1.0e6 << "/" << (int)total_bytes/1.0e6 << endl << endl;
    if(free_bytes > max_free) {
        max_free = free_bytes;
        selected = i;
    }
  }
  cerr << "[cnn] **USING DEVICE: " << selected << endl;
  CUDA_CHECK(cudaSetDevice(selected));
  device_id = selected;

  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
  CUDA_CHECK(cudaMalloc(&kSCALAR_MINUSONE, sizeof(cnn::real)));
  CUDA_CHECK(cudaMalloc(&kSCALAR_ONE, sizeof(cnn::real)));
  CUDA_CHECK(cudaMalloc(&kSCALAR_ZERO, sizeof(cnn::real)));
  cnn::real minusone = -1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_MINUSONE, &minusone, sizeof(cnn::real), cudaMemcpyHostToDevice));
  cnn::real one = 1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ONE, &one, sizeof(cnn::real), cudaMemcpyHostToDevice));
  cnn::real zero = 0;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ZERO, &zero, sizeof(cnn::real), cudaMemcpyHostToDevice));

  Initialize_Consts_And_Store_In_GPU();

  Initialize_CUDNN();
}

} // namespace cnn
