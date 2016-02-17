#pragma once

#include "cnn/macros.h"
#include "gpu-kernels.h"

namespace cnn {
namespace gpu {

    // each block processes one column. There must be 512 threads in a block
    template <class ElemType>
    __global__ void _assignColumnwiseLogSoftmaxOf(
        const ElemType* a,
        ElemType* us,
        const int m_numCols,
        const int m_numRows)
    {
        // We first find max per column
        __shared__ ElemType colMax[1];
        __shared__ ElemType partials[MAX_THREADS_PER_BLOCK];
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
        __shared__ ElemType colSum[1];
        colSum[0] = 0.0f;
        for (int i = threadIdx.x; i < m_numRows; i += MAX_THREADS_PER_BLOCK)
        {
            ElemType tmp = a[IDX2C(i, blockIdx.x, m_numRows)] - colMax[0];
            us[IDX2C(i, blockIdx.x, m_numRows)] = tmp;
            partials[threadIdx.x] += (sizeof(ElemType) == sizeof(float)) ? expf(tmp) : exp(tmp);
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
            colSum[0] = (sizeof(ElemType) == sizeof(float)) ? logf(colSum[0]) : log(colSum[0]);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < m_numRows; i += 512)
        {
            us[IDX2C(i, blockIdx.x, m_numRows)] -= colSum[0];
        }
    }

    // each block processes one column. There must be 512 threads in a block
    template <class ElemType>
    __global__ void _assignColumnwiseSoftmaxOf(
        const ElemType* a,
        ElemType* us,
        const int m_numCols,
        const int m_numRows)
    {
        // We first find max per column
        __shared__ ElemType colMax[1];
        __shared__ ElemType partials[MAX_THREADS_PER_BLOCK];
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
        __shared__ ElemType colSum[1];
        colSum[0] = 0.0f;
        for (int i = threadIdx.x; i < m_numRows; i += MAX_THREADS_PER_BLOCK)
        {
            ElemType tmp = a[IDX2C(i, blockIdx.x, m_numRows)] - colMax[0];
            us[IDX2C(i, blockIdx.x, m_numRows)] = tmp;
            partials[threadIdx.x] += (sizeof(ElemType) == sizeof(float)) ? expf(tmp) : exp(tmp);
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
            colSum[0] = (sizeof(ElemType) == sizeof(float)) ? logf(colSum[0]) : log(colSum[0]);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < m_numRows; i += 512)
        {
            ElemType tmp = us[IDX2C(i, blockIdx.x, m_numRows)] - colSum[0];
            us[IDX2C(i, blockIdx.x, m_numRows)] = (sizeof(ElemType) == sizeof(float)) ? expf(tmp) : exp(tmp);
        }
    }

    template <class ElemType>
    __global__ void _vectorSum(
        ElemType* c,       // output
        const ElemType* a, // input
        const int n, // a.numRows
        const int m, // a.numCols
        const bool isColWise)
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if ((isColWise && id >= m) || (!isColWise && id >= n))
            return;

        ElemType sum = 0;

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

    template <class ElemType>
    __global__ void _rowElementMultiplyWith(
        ElemType* us,
        const ElemType* a,
        const int N, // usrow;
        const int M) // acol;
    {
        int id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id >= M)
            return;

        // __shared__ ElemType _a[MAX_THREADS_PER_BLOCK];
        // _a[threadIdx.x]=a[id];
        ElemType mul = a[id];
        for (int i = 0; i < N; ++i)
        {
            us[IDX2C(i, id, N)] = us[IDX2C(i, id, N)] * mul;
        }
    }

    template <class ElemType>
    void AssignElementProductOf(int arow, int acol, const ElemType* a,
        int brow, int bcol, const ElemType* b,
        int crow, int ccol, ElemType* c)
    {
        assert(arow == brow && acol == bcol);
        if (!(arow == brow && acol == bcol))
            throw std::invalid_argument("The input matrix dimensions do not match.");

        if (!(crow == brow && ccol == bcol))
            throw std::invalid_argument("The input matrix dimensions do not match.");

        int N = (int)crow * ccol;
        int blocksPerGrid = (int)ceil(((double)N) / MAX_THREADS_PER_BLOCK);

        cudaEvent_t done = nullptr;
        cudaEventCreate(&done);
        _assignElementProductOf<ElemType> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(c, a, b, N);
        cudaEventRecord(done);
        cudaEventSynchronize(done);
        cudaEventDestroy(done);
    }

    template <class ElemType>
    void InnerProduct(unsigned arow, unsigned acol, const ElemType* a, 
        unsigned brow, unsigned bcol, const ElemType* b, 
        unsigned crow, unsigned ccol, ElemType* c, 
        const bool isColWise)
    {
        const int m = (int)arow; 
        const int n = (int)acol;
        const int k = (int)brow;
        const int l = (int)bcol;

        assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
        assert(m == k && n == l);                 // converting from size_t to int may cause overflow
        if (m != k || n != l)
            throw std::invalid_argument("Matrices a and b should have same dimension.");

        if (isColWise)
        {
            if (crow != 1 || ccol != n)
                throw std::invalid_argument("InnerProduct output matrix dimension mismatch");
        }
        else
        {
            if (crow != m || ccol != 1)
                throw std::invalid_argument("InnerProduct output matrix dimension mismatch");
        }

        if ((isColWise && m == 1) || !isColWise && n == 1) // in this case it's equivalent to element-wise product
        {
            AssignElementProductOf<ElemType>(arow, acol, a, brow, bcol, b, crow, ccol, c);
        }
        else
        {
            cudaEvent_t done = nullptr;

            int blocksPerGrid = 0;
            if (isColWise) // col-wise
            {
                blocksPerGrid = (int)ceil(1.0 * n / MAX_THREADS_PER_BLOCK);
            }
            else
            {
                blocksPerGrid = (int)ceil(1.0 * m / MAX_THREADS_PER_BLOCK);
            }

            cudaEventCreate(&done);
            _innerProduct<ElemType> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(c, a, b, m, n, isColWise);
            cudaEventRecord(done);
            cudaEventSynchronize(done);
            cudaEventDestroy(done);
        }
    }

    /// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + b</summary>
    /// if a is a column vector, add to all columns of b
    /// if a is a row vector, add to all rows of b
    /// if a is a scalar, add to all elements of b
    /// <param name="alpha">Scalar</param>
    /// <param name="a">Input matrix</param>
    /// <param name="b">Input matrix</param>
    /// <param name="c">Resulting matrix, user is responsible for allocating this</param>
    template <class ElemType>
    void ScaleAndAdd(ElemType alpha, int arow, int acol, const ElemType* a, int brow, int bcol, const ElemType* b,
        int crow, int ccol, ElemType* c)
    {
        return;

        if (crow != brow || ccol != bcol)
            throw std::invalid_argument("output dimension mismatch");

        if (arow == brow && acol == bcol) // dimensions match
        {

            unsigned N = crow * ccol;
            int blocksPerGrid = (int)ceil(1.0 * N / MAX_THREADS_PER_BLOCK);

            cudaEvent_t done = nullptr;
            cudaEventCreate(&done);
            _matrixMatrixAddOnCuda<ElemType> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(alpha, a, b, c, N);
            cudaEventRecord(done);
            cudaEventSynchronize(done);
            cudaEventDestroy(done);
        }
        else if (arow * acol == 1)
        {
            unsigned N = crow * ccol;
            int blocksPerGrid = (int)ceil(1.0 * N / MAX_THREADS_PER_BLOCK);

            cudaEvent_t done = nullptr;
            cudaEventCreate(&done);
            _scaleAndAddScalar<ElemType> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(c, N, alpha, a, b);
            cudaEventRecord(done);
            cudaEventSynchronize(done);
            cudaEventDestroy(done);
        }
        else if (acol == 1) // col vector, add it to all columns
        {
            unsigned m = (int)crow;
            unsigned n = (int)ccol;
            if (m != (int)arow)
                throw std::invalid_argument("To add column vector, rows should match.");

            cudaEvent_t done = nullptr;
            int blocksPerGrid = (int)(ceil(1.0 * m * n / MAX_THREADS_PER_BLOCK));
            cudaEventCreate(&done);
            _matrixVectorColumnWiseAddWithThreadPerElem<ElemType> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(a, b, c, alpha, m, n);

            cudaEventRecord(done);
            cudaEventSynchronize(done);
            cudaEventDestroy(done);
        }
        else if (arow == 1) // row vector, add it to all rows
        {
            int m = (int)crow;
            int n = (int)ccol;
            if (n != (int)acol)
                throw std::invalid_argument("To add row vector, columns should match.");
            cudaEvent_t done = nullptr;
            int blocksPerGrid = (int)(ceil(1.0 * m * n / MAX_THREADS_PER_BLOCK));
            cudaEventCreate(&done);
            _matrixVectorRowWiseAddWithThreadPerElem<ElemType> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(a, b, c, alpha, m, n);

            cudaEventRecord(done);
            cudaEventSynchronize(done);
            cudaEventDestroy(done);
        }
        else
            throw std::invalid_argument("dimension of matrix c does not match dimension of matrix a.");
    }

} // namespace gpu
} // namespace cnn

