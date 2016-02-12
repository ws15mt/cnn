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

} // namespace gpu
} // namespace cnn

#endif
