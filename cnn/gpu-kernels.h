#ifndef CNN_GPU_KERNELS_H
#define CNN_GPU_KERNELS_H

#include "cnn/cuda.h"
#include "macros.h"

namespace cnn {
    namespace gpu {

template<typename Func>
__global__ void unaryExprKernel(int n, const cnn::real* x, cnn::real* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void accUnaryExprKernel(int n, const cnn::real* x, cnn::real* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void binaryExprKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x0[i], x1[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void accBinaryExprKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y, Func func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        y[i] += func(x0[i], x1[i]);
        i += gridDim.x * blockDim.x;
    }
}

template<typename Func>
__global__ void accTripletExprKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real *x2, cnn::real* y, Func func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        y[i] += func(x0[i], x1[i], x2[i]);
        i += gridDim.x * blockDim.x;
    }
}

template<typename Func>
__global__ void slowReduceKernel(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y, Func func) {
  cnn::real ty = 0;
  // THIS IS BAD - FIX THIS TO MAKE IT FAST
  for (int i = 0; i < n; ++i)
    ty += func(x0[i], x1[i]);
  y[0] = ty;
}

// adapted from NVIDIA example
__global__ void ker_l2_norm_reducer(int n, const cnn::real *x0, cnn::real* res, bool sq, bool acc) {
    __shared__ cnn::real buf[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        cnn::real sum = 0;
        for (int pos = i; pos < n; pos += 256) {
            const cnn::real d = x0[pos];
            sum += sq ? d * d : d;
        }
        buf[i] = sum;
    }
    for (int stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        for (int i = threadIdx.x; i < stride; i += blockDim.x)
            buf[i] += buf[stride + i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (acc) res[0] += buf[0]; else res[0] = buf[0];
    }
}

// A kernel to calculate the dot product between two arrays
__global__ void ker_dotproduct(int n, const cnn::real* x, const cnn::real* y, cnn::real* z) {
    __shared__ cnn::real buf[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        cnn::real sum = 0;
        for (int pos = i; pos < n; pos += 256)
            sum += x[pos] * y[pos];
        buf[i] = sum;
    }
    for (int stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        for (int i = threadIdx.x; i < stride; i += blockDim.x)
            buf[i] += buf[stride + i];
    }
    __syncthreads();
    if (threadIdx.x == 0)
        z[0] = buf[0];
}

// adapted from NVIDIA example
__global__ void ker_sqeucdist(int n, const cnn::real *x0, const cnn::real *x1, cnn::real* res) {
    __shared__ cnn::real buf[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        cnn::real sum = 0;
        for (int pos = i; pos < n; pos += 256) {
            const cnn::real d = x0[pos] - x1[pos];
            sum += d * d;
        }
        buf[i] = sum;
    }
    for (int stride = 128; stride > 0; stride >>= 1) {
        __syncthreads();
        for (int i = threadIdx.x; i < stride; i += blockDim.x)
            buf[i] += buf[stride + i];
    }
    __syncthreads();
    if (threadIdx.x == 0) res[0] = buf[0];
}


} // namespace gpu
} // namespace cnn

#endif
