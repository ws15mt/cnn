#ifndef CNN_GPU_KERNELS_H
#define CNN_GPU_KERNELS_H

#include "cnn/cuda.h"
#include "macros.h"

namespace cnn {
    namespace gpu {

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
    unsigned id = blockDim.x * blockIdx.x + threadIdx.x;   \
    if (id >= N)                                   \
    return;

template<typename Func>
__global__ void unaryExprKernel(int n, const float* x, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void accUnaryExprKernel(int n, const float* x, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += func(x[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void binaryExprKernel(int n, const float* x0, const float* x1, float* y, Func func) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = func(x0[i], x1[i]);
    i += gridDim.x * blockDim.x;
  }
}

template<typename Func>
__global__ void accBinaryExprKernel(int n, const float* x0, const float* x1, float* y, Func func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        y[i] += func(x0[i], x1[i]);
        i += gridDim.x * blockDim.x;
    }
}

template<typename Func>
__global__ void accTripletExprKernel(int n, const float* x0, const float* x1, float *x2, float* y, Func func) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < n) {
        y[i] += func(x0[i], x1[i], x2[i]);
        i += gridDim.x * blockDim.x;
    }
}

template<typename Func>
__global__ void slowReduceKernel(int n, const float* x0, const float* x1, float* y, Func func) {
  float ty = 0;
  // THIS IS BAD - FIX THIS TO MAKE IT FAST
  for (int i = 0; i < n; ++i)
    ty += func(x0[i], x1[i]);
  y[0] = ty;
}

template <class ElemType>
__global__ void _innerProduct(
    ElemType* c,
    const ElemType* a,
    const ElemType* b,
    const int N, // a.GetNumRows();
    const int M, // a.GetNumCols();
    const bool isColWise)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if ((isColWise && id >= M) || (!isColWise && id >= N))
        return;

    ElemType sum = 0;
    int index;
    if (isColWise)
    {
        for (int i = 0; i < N; ++i)
        {
            index = IDX2C(i, id, N);
            sum += a[index] * b[index];
        }
    }
    else
    {
        for (int j = 0; j < M; ++j)
        {
            index = IDX2C(id, j, N);
            sum += a[index] * b[index];
        }
    }

    c[id] = sum;
}

template <class ElemType>
__global__ void _assignElementProductOf(
    ElemType* us,
    const ElemType* a,
    const ElemType* b,
    const int N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    us[id] = a[id] * b[id];
}

///a is a scalar
template <class ElemType>
__global__ void _scaleAndAddScalar(
    ElemType* c,
    const int N,
    const ElemType alpha,
    const ElemType* a,
    const ElemType* b)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= N)
        return;
    c[id] = alpha * a[0] + b[id];
};

template <class ElemType>
__global__ void _matrixMatrixAddOnCuda(
    const ElemType alpha,
    const ElemType* a,
    const ElemType* b,
    ElemType* c,
    const int N)
{
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
    c[id] = alpha * a[id] + b[id];
}

//this implementation uses more threads but also more memory access
template <class ElemType>
__global__ void _matrixVectorColumnWiseAddWithThreadPerElem(
    const ElemType* a,
    const ElemType* b,
    ElemType* us,
    ElemType alpha,
    const int m, // number of rows
    const int n) // number of cols
{
    int N = m * n; // used in CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id,N) macro
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

    int col = id / m;
    int row = id - col * m;

    us[id] = alpha * a[row] + b[id];
}

template <class ElemType>
__global__ void _matrixVectorRowWiseAddWithThreadPerElem(
    const ElemType* a,
    const ElemType* b,
    ElemType* us,
    ElemType alpha,
    const int m, // number of rows
    const int n) // number of cols
{
    int N = m * n; // used in CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id,N) macro
    CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

    int col = id / m;

    us[id] = alpha * a[col] + b[id];
}

// adapted from NVIDIA example
__global__ void ker_l2_norm_reducer(int n, const float *x0, float* res, bool sq, bool acc) {
    __shared__ float buf[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float sum = 0;
        for (int pos = i; pos < n; pos += 256) {
            const float d = x0[pos];
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
__global__ void ker_dotproduct(int n, const float* x, const float* y, float* z) {
    __shared__ float buf[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float sum = 0;
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
__global__ void ker_sqeucdist(int n, const float *x0, const float *x1, float* res) {
    __shared__ float buf[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float sum = 0;
        for (int pos = i; pos < n; pos += 256) {
            const float d = x0[pos] - x1[pos];
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
