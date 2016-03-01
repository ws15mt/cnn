#pragma once

#include "cnn/macros.h"
#include "gpu-kernels.h"

namespace cnn {
namespace gpu {

    // each block processes one column. There must be 512 threads in a block
    template <class T>
    __global__ void _assignColumnwiseLogSoftmaxOf(
        const T* a,
        T* us,
        const int m_numCols,
        const int m_numRows)
    {
        // We first find max per column
        __shared__ T colMax[1];
        __shared__ T partials[MAX_THREADS_PER_BLOCK];
        colMax[0] = -10000000;
        partials[threadIdx.x] = -10000000;

        for (int i = threadIdx.x; i < m_numRows; i += MAX_THREADS_PER_BLOCK)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x], a[IDX2C(i, blockIdx.x, m_numRows)]);
        }
        __syncthreads();

        if (threadIdx.x < 256)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 256], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 128)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 128], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 64)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 64], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 32)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 32], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 16)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 16], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 8)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 8], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 4)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 4], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            colMax[0] = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
        }
        partials[threadIdx.x] = 0.0f;
        __syncthreads();

        // Now start finding sums
        __shared__ T colSum[1];
        colSum[0] = 0.0f;
        for (int i = threadIdx.x; i < m_numRows; i += MAX_THREADS_PER_BLOCK)
        {
            T tmp = a[IDX2C(i, blockIdx.x, m_numRows)] - colMax[0];
            us[IDX2C(i, blockIdx.x, m_numRows)] = tmp;
            partials[threadIdx.x] += (sizeof(T) == sizeof(float)) ? expf(tmp) : exp(tmp);
        }
        __syncthreads();

        if (threadIdx.x < 256)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 256];
        }
        __syncthreads();

        if (threadIdx.x < 128)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 128];
        }
        __syncthreads();

        if (threadIdx.x < 64)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 64];
        }
        __syncthreads();

        if (threadIdx.x < 32)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 32];
        }
        __syncthreads();

        if (threadIdx.x < 16)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 16];
        }
        __syncthreads();

        if (threadIdx.x < 8)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 8];
        }
        __syncthreads();

        if (threadIdx.x < 4)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 4];
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            colSum[0] = partials[0] + partials[1] + partials[2] + partials[3];
            colSum[0] = (sizeof(T) == sizeof(float)) ? logf(colSum[0]) : log(colSum[0]);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < m_numRows; i += 512)
        {
            us[IDX2C(i, blockIdx.x, m_numRows)] -= colSum[0];
        }
    }

    // each block processes one column. There must be 512 threads in a block
    template <class T>
    __global__ void _assignColumnwiseSoftmaxOf(
        const T* a,
        T* us,
        const int m_numCols,
        const int m_numRows)
    {
        // We first find max per column
        __shared__ T colMax[1];
        __shared__ T partials[MAX_THREADS_PER_BLOCK];
        colMax[0] = -10000000;
        partials[threadIdx.x] = -10000000;

        for (int i = threadIdx.x; i < m_numRows; i += MAX_THREADS_PER_BLOCK)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x], a[IDX2C(i, blockIdx.x, m_numRows)]);
        }
        __syncthreads();

        if (threadIdx.x < 256)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 256], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 128)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 128], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 64)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 64], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 32)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 32], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 16)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 16], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 8)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 8], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x < 4)
        {
            partials[threadIdx.x] = max(partials[threadIdx.x + 4], partials[threadIdx.x]);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            colMax[0] = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
        }
        partials[threadIdx.x] = 0.0f;
        __syncthreads();

        // Now start finding sums
        __shared__ T colSum[1];
        colSum[0] = 0.0f;
        for (int i = threadIdx.x; i < m_numRows; i += MAX_THREADS_PER_BLOCK)
        {
            T tmp = a[IDX2C(i, blockIdx.x, m_numRows)] - colMax[0];
            us[IDX2C(i, blockIdx.x, m_numRows)] = tmp;
            partials[threadIdx.x] += (sizeof(T) == sizeof(float)) ? expf(tmp) : exp(tmp);
        }
        __syncthreads();

        if (threadIdx.x < 256)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 256];
        }
        __syncthreads();

        if (threadIdx.x < 128)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 128];
        }
        __syncthreads();

        if (threadIdx.x < 64)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 64];
        }
        __syncthreads();

        if (threadIdx.x < 32)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 32];
        }
        __syncthreads();

        if (threadIdx.x < 16)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 16];
        }
        __syncthreads();

        if (threadIdx.x < 8)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 8];
        }
        __syncthreads();

        if (threadIdx.x < 4)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 4];
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            colSum[0] = partials[0] + partials[1] + partials[2] + partials[3];
            colSum[0] = (sizeof(T) == sizeof(float)) ? logf(colSum[0]) : log(colSum[0]);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < m_numRows; i += 512)
        {
            T tmp = us[IDX2C(i, blockIdx.x, m_numRows)] - colSum[0];
            us[IDX2C(i, blockIdx.x, m_numRows)] = (sizeof(T) == sizeof(float)) ? expf(tmp) : exp(tmp);
        }
    }

    // each block processes one column. There must be 512 threads in a block
    template <class T>
    __global__ void _assignColumnwiseSoftmaxOfBackward(
        const T* fx,   /// value of this softmax
        const T* dEdf, /// gradient to be backpropagated
        T* dEdxi,      /// backpropagated gradient
        const int m_numCols,
        const int m_numRows)
    {
        __shared__ T partials[MAX_THREADS_PER_BLOCK];
        partials[threadIdx.x] = 0.0f; 
        __shared__ T innerProductAtThisColumn[1];
        innerProductAtThisColumn[0] = 0.0f;

        for (int i = threadIdx.x; i < m_numRows; i += MAX_THREADS_PER_BLOCK)
        {
            T v = fx[IDX2C(i, blockIdx.x, m_numRows)];
            T tmp = v * dEdf[IDX2C(i, blockIdx.x, m_numRows)]; 
            dEdxi[IDX2C(i, blockIdx.x, m_numRows)] += tmp;  /// elementwise product of value and gradient, and add it to the input gradient
            partials[threadIdx.x] += tmp;
        }
        __syncthreads();

        if (threadIdx.x < 256)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 256];
        }
        __syncthreads();

        if (threadIdx.x < 128)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 128];
        }
        __syncthreads();

        if (threadIdx.x < 64)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 64];
        }
        __syncthreads();

        if (threadIdx.x < 32)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 32];
        }
        __syncthreads();

        if (threadIdx.x < 16)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 16];
        }
        __syncthreads();

        if (threadIdx.x < 8)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 8];
        }
        __syncthreads();

        if (threadIdx.x < 4)
        {
            partials[threadIdx.x] += partials[threadIdx.x + 4];
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            innerProductAtThisColumn[0] = partials[0] + partials[1] + partials[2] + partials[3];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < m_numRows; i += 512)
        {
            T tmp = innerProductAtThisColumn[0] * fx[IDX2C(i, blockIdx.x, m_numRows)];
            dEdxi[IDX2C(i, blockIdx.x, m_numRows)] -= tmp; /// subtract with inner product of this column times softmax value
        }
    }

    template <class T>
    __global__ void _vector_sum(
        T* c,       // output
        const T* a, // input
        const int n, // a.numRows
        const int m, // a.numCols
        const bool isColWise)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if ((isColWise && id >= m) || (!isColWise && id >= n))
            return;

        T sum = 0;

        if (isColWise)
        {
            for (int i = 0; i < n; ++i)
            {
                sum += a[IDX2C(i, id, n)];
            }
        }
        else
        {
            for (int j = 0; j < m; ++j)
            {
                sum += a[IDX2C(id, j, n)];
            }
        }
        c[id] = sum;
    }

    template <class T>
    __global__ void _vector_add_const(
        T* c,       // output
        const T* a, // input
        const int n, // a.numRows
        const int m, // a.numCols
        const T* b,  // the const value
        const bool isColWise)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if ((isColWise && id >= m) || (!isColWise && id >= n))
            return;
        __shared__ T s[1];
        s[0] = *b;
        __syncthreads();

        c[id] = s[0] + a[id];
    }

    template <class T>
    __global__ void _rowElementMultiplyWith(
        T* us,
        const T* a,
        const int N, // usrow;
        const int M) // acol;
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id >= M)
            return;

        // __shared__ T _a[MAX_THREADS_PER_BLOCK];
        // _a[threadIdx.x]=a[id];
        T mul = a[id];
        for (int i = 0; i < N; ++i)
        {
            us[IDX2C(i, id, N)] = us[IDX2C(i, id, N)] * mul;
        }
    }

} // namespace gpu
} // namespace cnn

