#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#include "cnn/gpu-kernels.h"
#include "cnn/functors.h"
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "gpu-ops.cuh"

namespace cnn {
namespace gpu {

// this wraps kernel dispatches for various operations (preventing us from
// having to compile a version of nodes.cc with NVCC)

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void set_to_value_of(int n, float* x0, float val)
{
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(x0);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + n, val);
}

void set_to_value_of(int n, float* x0, float *val) {
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(x0);
    thrust::device_ptr<float> src_dev_ptr = thrust::device_pointer_cast(val);
    thrust::copy(src_dev_ptr, src_dev_ptr + n, dev_ptr);
}

void vpairwise_rank_loss(int n, float margin, const float* xgood, const float* xbad, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprKernel<<<tb.first, tb.second>>>(n, xgood, xbad, y, FPairwiseRankLoss(margin));
}

void vpairwise_rank_loss_backward(int n, bool d_wrt_correct, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  if (d_wrt_correct) {
    accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyNegateBackward());
  } else {
    accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyBackward());
  }
}

void vcwise_product(int n, const float* x0, const float* x1, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprKernel<<<tb.first, tb.second>>>(n, x0, x1, y, FProduct());
}

void vcwise_product_backward(int n, const float* dEdy, const float* x_other, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, dEdy, x_other, dEdx, FProduct());
}

void vcwise_quotient(int n, const float* x0, const float* x1, float* y) {
    auto tb = SizeToBlockThreadPair(n);
    binaryExprKernel << <tb.first, tb.second >> >(n, x0, x1, y, FQuotient());
}

void vcwise_quotient_backward(int n, const float* dEdy, const float* x_other, float* dEdx) {
    auto tb = SizeToBlockThreadPair(n);
    accBinaryExprKernel << <tb.first, tb.second >> >(n, dEdy, x_other, dEdx, FQuotient());
}

void vconstant_minusx(int n, float c, const float* x, float* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FConstantMinus(c));
}

void vconstant_multiplyx(int n, float c, const float* x, float* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FConstantMultiply(c));
}

void vconstant_multiplyx_backward(int n, float c, const float* x, float* y) {
    auto tb = SizeToBlockThreadPair(n);
    accUnaryExprKernel << <tb.first, tb.second >> >(n, x, y, FConstantMultiply(c));
}

void vexp(int n, const float* x, float* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FExp());
}

void vnegate(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FNegate());
}

void vnegate_backward(int n, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accUnaryExprKernel<<<tb.first, tb.second>>>(n, dEdf, dEdx, FNegate());
}

void vrelu(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FRectify());
}

void vrelu_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyBackward());
}

void vtanh(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FTanh());
}

void vtanh_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FTanhBackward());
}

void vlog(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FLog());
}

void vlog_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FLogBackward());
}

void vlogistic(int n, const float* x, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FLogisticSigmoid());
}

void vlogistic_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FLogisticSigmoidBackward());
}

void sqeucdist_backward(int n, const float* dEdy, const float* x0, const float* x1, float* dEdx, int i) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, x0, x1, dEdx, FEuclideanBackward(i, dEdy));
}

void sgd_update(int n, const float* g, float* x, float scale, float lambda) {
    auto tb = SizeToBlockThreadPair(n);
    accBinaryExprKernel << <tb.first, tb.second >> >(n, x, g, x, FL2SGDUpdate(lambda, scale));
}

void sgd_momentum_update(int n, const float* g, float* x, float* v, float scale, float lambda, float momentum) {
    auto tb = SizeToBlockThreadPair(n);
    accTripletExprKernel << <tb.first, tb.second >> >(n, x, g, v, x, FL2SGDMomentumUpdate(lambda, scale, momentum));
}

/** followed some examples of using thrust at
https://github.com/OrangeOwlSolutions/Thrust/blob/master/Calculating_the_norm_of_arrays.cu
*/
void rmsprop_momentum_update(int n, const float* g, float* x, float* v, float *r, float scale, float lambda, float momentum, float rho, float epsilon) {
    auto tb = SizeToBlockThreadPair(n);
    float squared_norm = thrust::transform_reduce(thrust::device_pointer_cast(g), thrust::device_pointer_cast(g + n), FSquare(), (float)0.0, thrust::plus<float>());
    *r = rho * (*r) + (1 - rho) * squared_norm;
    float den = sqrt(*r + epsilon);
    accTripletExprKernel << <tb.first, tb.second >> >(n, x, g, v, x, FL2SGDMomentumUpdate(lambda, scale / den, momentum));
    //CUDA_CHECK(cudaFree(sqnorm));
}

void sqeucdist(int n, const float* x0, const float *x1, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  ker_sqeucdist<<<tb.first,tb.second>>>(n, x0, x1, y);
}

void l2_norm_reducer(int n, const float* x0, float* y, bool square, bool accumulate) {
  auto tb = SizeToBlockThreadPair(n);
  ker_l2_norm_reducer<<<tb.first,tb.second>>>(n, x0, y, square, accumulate);
}

void VectorSum(int rows, int cols, const float * a, float* c, const bool isColWise)
{
    assert(rows > 0 && cols > 0); // converting from size_t to int may cause overflow

    int m = cols;
    int n = rows;

    cudaEvent_t done = nullptr;

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        blocksPerGrid = (int)ceil(1.0 * m / MAX_THREADS_PER_BLOCK);
    }
    else
    {
        blocksPerGrid = (int)ceil(1.0 * n / MAX_THREADS_PER_BLOCK);
    }

    cudaEventCreate(&done);
    _vectorSum<float> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(c, a, n, m, isColWise);
    cudaEventRecord(done);
    cudaEventSynchronize(done);
    cudaEventDestroy(done);
}

/// assume that a is a vector with col dimension
void RowElementMultiplyWith(int arow, int acol, const float * a, int brow, int bcol, float * b)
{
    if (arow != 1 || acol != bcol)
    {
        abort();
    }

    int N = brow; 
    int M = acol;
    int blocksPerGrid = (int)ceil(1.0 * M / MAX_THREADS_PER_BLOCK);

    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);
    _rowElementMultiplyWith<float> << <blocksPerGrid, MAX_THREADS_PER_BLOCK >> >(b, a, N, M);
    cudaEventRecord(done);
    cudaEventSynchronize(done);
    cudaEventDestroy(done);
}

void logsoftmax(int row, int col, const float* x0, float* y) 
{
    cudaStream_t t_stream = cudaStreamDefault;

    int N = col;
    int M = row;
    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);
    _assignColumnwiseLogSoftmaxOf<float> << <N, 512, 0, t_stream >> >(x0, y, N, M);
    
    cudaEventRecord(done);
    
    cudaEventSynchronize(done);
    
    cudaEventDestroy(done);
}

void logsoftmax_backward(int row, int col, const float *fx, const float *dEdf, float *dEdx, float * gpu_softmax, float *grd)
{
    vexp(row * col, fx, gpu_softmax);
    VectorSum(row, col, dEdf, grd, true); 
    RowElementMultiplyWith(1, col, grd, row, col, gpu_softmax);

    auto tb = SizeToBlockThreadPair(col * row);
    accBinaryExprKernel << <tb.first, tb.second >> >(col * row, dEdf, gpu_softmax, dEdx, FSubtract());
}

void softmax(int row, int col, const float* x0, float* y)
{
    cudaStream_t t_stream = cudaStreamDefault;

    int N = col;
    int M = row;
    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);
    _assignColumnwiseSoftmaxOf<float> << <N, MAX_THREADS_PER_BLOCK, 0, t_stream >> >(x0, y, N, M);

    cudaEventRecord(done);

    cudaEventSynchronize(done);

    cudaEventDestroy(done);
}

/// see http://research.microsoft.com/pubs/226641/CNTKBook-20160121.pdf
void softmax_backward(int row, int col, const float *fx, const float *dEdf, float *dEdx, float *tmp_one_row)
{
    int n = row * col;
    auto tb = SizeToBlockThreadPair(n);
    float ods;
    ker_dotproduct << <tb.first, tb.second >> >(n, fx, dEdf, tmp_one_row);
    cudaMemcpy(&ods, tmp_one_row, sizeof(float), cudaMemcpyDeviceToHost);
    accBinaryExprKernel << <tb.first, tb.second >> >(n, fx, dEdf, dEdx, FSoftmaxBackward(-ods));
}

// adapted from NVIDIA example
__global__ void ker_pnlsoftmax(int n, int elem_idx, const float *x0, float* res, float* logz) {
  __shared__ float buf[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    float me = __int_as_float(0xff800000);
    for (int pos = i; pos < n; pos += 256) {
      const float d = x0[pos];
      me = d > me ? d : me;
    }
    buf[i] = me;
  }
  for (int stride = 128; stride > 0; stride >>= 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < stride; i += blockDim.x)
        buf[i] = buf[i] > buf[stride + i] ? buf[i] : buf[stride + i];
  }
  __syncthreads();
  const float max_elem = buf[0];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    float sum = 0;
    for (int pos = i; pos < n; pos += 256)
      sum += expf(x0[pos] - max_elem);
    buf[i] = sum;
  }
  for (int stride = 128; stride > 0; stride >>= 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < stride; i += blockDim.x)
        buf[i] += buf[stride + i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float lz = log(buf[0]) + max_elem;
    logz[0] = lz;
    res[0] = lz - x0[elem_idx];
  }
}

void pnlsoftmax(int n, int elem_idx, const float* x0, float* y, float* logz) {
  auto tb = SizeToBlockThreadPair(n);
  ker_pnlsoftmax<<<tb.first,tb.second>>>(n, elem_idx, x0, y, logz);
}

__global__ void fixup_pnl(const float* dEdf, float* dEdxi, int i) {
  if (threadIdx.x == 0) dEdxi[i] -= dEdf[0];
}

void pnlsoftmax_backward(int n, int elem_idx, const float* x0, const float* dEdf, const float* logz, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accUnaryExprKernel<<<tb.first, tb.second>>>(n, x0, dEdx, FPtrNegLogSoftmaxBackward(logz, dEdf));
  fixup_pnl<<<1,1>>>(dEdf, dEdx, elem_idx);
}


void conv1dwide(const int n, const int m, const float* xs, const int k, const float *fx, float *fy)
{

    thrust::device_vector<float> dv((m + k) * n, 0.0);
    thrust::device_ptr<float> vp = dv.data();
    thrust::device_ptr<float> fp((float*)fx);
    thrust::device_ptr<float> xp((float*)xs);
    thrust::device_ptr<float> yp(fy);

    for (size_t tk = 0; tk < k; tk++)
    {
        for (size_t j = 0; j < m; j++)
            thrust::transform(xp + j * n, xp + (j + 1) * n, fp + tk * n, vp + tk * n + j * n, thrust::multiplies<float>());
    }
    thrust::copy(vp, vp + (m + k) * n, thrust::device_pointer_cast(fy));
}

void conv1dwide_backward(const int i, const int n, const int m, const float* xs, const int k, const float *fx, const float* dEdf, float *dEdx)
{
    thrust::device_vector<float> dv(m  * n, 0.0);
    thrust::device_ptr<float> vp = dv.data();
    thrust::device_ptr<float> fp((float*)fx);
    thrust::device_ptr<float> xp((float*)xs);
    thrust::device_ptr<float> d((float*)dEdf);
    thrust::device_ptr<float> yp(dEdx);

    for (size_t tk = 0; tk < k; tk++)
    {
        if (i == 0) { // derivative wrt input x
            for (size_t j = 0; j < m; j++)
                thrust::transform(d + j * n + tk*n, d + (j + 1) * n + tk*n, fp + tk * n, dv.data() + j * n, thrust::multiplies<float>());
        }
        else { // derivative wrt filter f
            for (size_t j = 0; j < m; j++)
                thrust::transform(d + j * n + tk*n, d + (j + 1) * n + tk*n, xp + j * n, dv.data() + tk * n, thrust::multiplies<float>());
        }
    }
    if (i == 0)
        thrust::transform(dv.data(), dv.data() + m * n, yp, yp, thrust::plus<float>());
    else 
        thrust::transform(dv.data(), dv.data() + k * n, yp, yp, thrust::plus<float>());
}

void addVectorToAllColumns(const int n, const float * xs, const int m, const float* fx, float *fy)
{
    thrust::device_ptr<float> fp((float*)fx);
    thrust::device_ptr<float> xp((float*)xs);
    thrust::device_ptr<float> yp(fy);
    for (size_t j = 0; j < n / m; j++)
        thrust::transform(xp + j * m, xp + (j + 1) * m, fp, yp + j * m, thrust::plus<float>());
}

void addVectorToAllColumns_backward(const int i, const int r, const int c, const float* dEdf, float *dEdxi)
{
    thrust::device_ptr<const float> dp(dEdf);
    thrust::device_ptr<float> dx(dEdxi);

    if (i == 0)
    {
        // x
        thrust::transform(dp, dp + r * c, dx, dx, thrust::plus<float>());
    }
    else
    {
        // bias
        for (int k = 0; k < c; k++)
            thrust::transform(dp + k * r, dp + (k + 1)*r, dx, dx, thrust::plus<float>());
    }
}

/**
stride : the jump step
*/
void foldRows(const int n, const int m, const float *xs, const int stride, const int orows, float *fy)
{
    thrust::device_ptr<float> xp((float*)xs), pp;
    thrust::device_ptr<float> yp(fy);
    thrust::host_vector<float> vo(orows * m);

    pp = xp;
    for (size_t j = 0; j < m; j++)
    {
        for (size_t r = 0; r < orows; r++)
        {
            vo[j * orows + r] = thrust::reduce(pp, pp + stride);
            pp += stride;
        }
    }
}

void foldRows_backward(const int orows, const float* dEdf, const int n, const int m, float *fy)
{
    thrust::device_ptr<float> dp((float*)dEdf);
    thrust::device_ptr<float> yp(fy);

    for (int i = 0; i < orows; ++i)
    {
        int stride = n / orows;
        for (int j = 0; j < m; j++)
        { // loop over columns
            for (int k = 0; k < stride; k++)
            {
                *(yp + i * stride + k + j * n) += *(dp + i + j * n);
            }
        }
    }
}

void kMaxPooling(const int n, const int m, const float *xs, const int k, float *fy, int* aux_mem)
{
    thrust::device_ptr<float> xp((float*)xs), pp;
    thrust::device_ptr<float> yp(fy);
    thrust::device_vector<float> vo(m);
    thrust::device_vector<float> vp(k);

    pp = xp;

    int* maxmap = static_cast<int*>(aux_mem);
    size_t mi = 0;
    for (unsigned i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; j++)
            vo[j] = (*(pp + i + j * n));
        thrust::sort(thrust::device, vo.data(), vo.data() + m);

        size_t mk = 0;
        for (int j = 0; j < m; j++)
        {
            if (mk == k)
                break;
            if (*(pp + i + j * n) >= vo[m - k])
            {
                *(yp + i + mk * n) = *(pp + i + j*n);
                cudaMemcpy(&maxmap[mi], &j, sizeof(int), cudaMemcpyHostToDevice); 
                mi++;
                mk++;
            }
        }
    }
}

void kMaxPooling_backward(const int n, const int m, const float *xs, const int k, const float * dEdf, float *dEdxi, const int* aux_mem)
{
    const int* maxmap = aux_mem;
    int mk = 0;
    int oj;
    thrust::device_ptr<const float> xp(xs);
    thrust::device_ptr<const float> dp(dEdf);
    thrust::device_ptr<float> yp(dEdxi);
    thrust::host_vector<int> hv(n, 0);
    cudaMemcpy(hv.data(), maxmap, sizeof(int)*n, cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            oj = hv[mk++];
            if (oj < k && oj >= 0){
                thrust::transform(dp + i + j * n, dp + i + j * n + 1, yp + i + oj * n, yp + i + oj * n, thrust::plus<float>());
            }
        }
    }
}


} // namespace gpu
} // namespace cnn
