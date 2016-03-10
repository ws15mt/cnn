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

void saxpy_fast(cnn::real A, thrust::device_vector<cnn::real>& X, thrust::device_vector<cnn::real>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void set_to_value_of(int n, cnn::real* x0, cnn::real val)
{
    thrust::device_ptr<cnn::real> dev_ptr = thrust::device_pointer_cast(x0);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + n, val);
}

void set_to_value_of(int n, cnn::real* x0, cnn::real *val) {
    thrust::device_ptr<cnn::real> dev_ptr = thrust::device_pointer_cast(x0);
    thrust::device_ptr<cnn::real> src_dev_ptr = thrust::device_pointer_cast(val);
    thrust::copy(src_dev_ptr, src_dev_ptr + n, dev_ptr);
}

void vpairwise_rank_loss(int n, cnn::real margin, const cnn::real* xgood, const cnn::real* xbad, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprKernel<<<tb.first, tb.second>>>(n, xgood, xbad, y, FPairwiseRankLoss(margin));
}

void vpairwise_rank_loss_backward(int n, bool d_wrt_correct, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  if (d_wrt_correct) {
    accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyNegateBackward());
  } else {
    accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyBackward());
  }
}

void vcwise_product(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  binaryExprKernel<<<tb.first, tb.second>>>(n, x0, x1, y, FProduct());
}

void vcwise_product_backward(int n, const cnn::real* dEdy, const cnn::real* x_other, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, dEdy, x_other, dEdx, FProduct());
}

void vcwise_quotient(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y) {
    auto tb = SizeToBlockThreadPair(n);
    binaryExprKernel << <tb.first, tb.second >> >(n, x0, x1, y, FQuotient());
}

void vcwise_quotient_backward(int n, const cnn::real* dEdy, const cnn::real* x_other, cnn::real* dEdx) {
    auto tb = SizeToBlockThreadPair(n);
    accBinaryExprKernel << <tb.first, tb.second >> >(n, dEdy, x_other, dEdx, FQuotient());
}

void vconstant_minusx(int n, cnn::real c, const cnn::real* x, cnn::real* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FConstantMinus(c));
}

void vconstant_minusx_backward(int n, cnn::real c, const cnn::real* x, cnn::real* y) {
    auto tb = SizeToBlockThreadPair(n);
    accUnaryExprKernel << <tb.first, tb.second >> >(n, x, y, FConstantMinus(c));
}

void vconstant_multiplyx(int n, cnn::real c, const cnn::real* x, cnn::real* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FConstantMultiply(c));
}

void vconstant_multiplyx_backward(int n, cnn::real c, const cnn::real* x, cnn::real* y) {
    auto tb = SizeToBlockThreadPair(n);
    accUnaryExprKernel << <tb.first, tb.second >> >(n, x, y, FConstantMultiply(c));
}

void vexp(int n, const cnn::real* x, cnn::real* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FExp());
}

void vnegate(int n, const cnn::real* x, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FNegate());
}

void vnegate_backward(int n, const cnn::real* dEdf, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accUnaryExprKernel<<<tb.first, tb.second>>>(n, dEdf, dEdx, FNegate());
}

void vrelu(int n, const cnn::real* x, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FRectify());
}

void vrelu_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FRectifyBackward());
}

void vexponential_linear_units(int n, const cnn::real* x, const cnn::real scale, cnn::real* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FExponentialLinearUnits(scale));
}

void vexponential_linear_units_backward(int n, const cnn::real* fx, const cnn::real* dEdf, const cnn::real scale, cnn::real* dEdx) {
    auto tb = SizeToBlockThreadPair(n);
    accBinaryExprKernel << <tb.first, tb.second >> >(n, fx, dEdf, dEdx, FExponentialLinearUnitsBackward(scale));
}

void vtanh(int n, const cnn::real* x, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FTanh());
}

void vtanh_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FTanhBackward());
}

void vlog(int n, const cnn::real* x, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FLog());
}

void vlog_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FLogBackward());
}

void vlogistic(int n, const cnn::real* x, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FLogisticSigmoid());
}

void vlogistic_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FLogisticSigmoidBackward());
}

void sqeucdist_backward(int n, const cnn::real* dEdy, const cnn::real* x0, const cnn::real* x1, cnn::real* dEdx, int i) {
  auto tb = SizeToBlockThreadPair(n);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, x0, x1, dEdx, FEuclideanBackward(i, dEdy));
}

void sgd_update(int n, const cnn::real* g, cnn::real* x, cnn::real scale, cnn::real lambda) {
    auto tb = SizeToBlockThreadPair(n);
    accBinaryExprKernel << <tb.first, tb.second >> >(n, x, g, x, FL2SGDUpdate(lambda, scale));
}

void sgd_update(int n, const cnn::real* g, cnn::real* x, cnn::real* scale, cnn::real* lambda) {
    auto tb = SizeToBlockThreadPair(n);
    accBinaryExprKernel << <tb.first, tb.second >> >(n, x, g, x, FL2SGDUpdatePtrArguments (lambda, scale));
}

void sgd_momentum_update(int n, const cnn::real* g, cnn::real* x, cnn::real* v, cnn::real scale, cnn::real lambda, cnn::real momentum) {
    auto tb = SizeToBlockThreadPair(n);
    accTripletExprKernel << <tb.first, tb.second >> >(n, x, g, v, x, FL2SGDMomentumUpdate(lambda, scale, momentum));
}

/** followed some examples of using thrust at
https://github.com/OrangeOwlSolutions/Thrust/blob/master/Calculating_the_norm_of_arrays.cu
*/
/// this is old code that computes gradient norm for every parameter
/*
void rmsprop_momentum_update(int n, const cnn::real* g, cnn::real* x, cnn::real* v, cnn::real *r, cnn::real scale, cnn::real lambda, cnn::real momentum, cnn::real rho, cnn::real epsilon) {
    auto tb = SizeToBlockThreadPair(n);
    /// it may be more efficient to compute in cpu and not do reduce in gpu, but my observation is not 
    /// that case
    cnn::real squared_norm = thrust::transform_reduce(thrust::device_pointer_cast(g), thrust::device_pointer_cast(g + n), FSquare(), (cnn::real)0.0, thrust::plus<cnn::real>());
    *r = rho * (*r) + (1 - rho) * squared_norm;
    cnn::real den = sqrt(*r + epsilon);
    accTripletExprKernel << <tb.first, tb.second >> >(n, x, g, v, x, FL2SGDMomentumUpdate(lambda, scale / den, momentum));
    //CUDA_CHECK(cudaFree(sqnorm));
}
*/

/// this is a newer code that uses gradient norms computed elsewhere. 
/// potential speed-up can be achieved to compute all of gradient norms in GPU and then transfer them to
/// CPU in a bulk. 
void rmsprop_momentum_update(int n, const cnn::real* g, cnn::real* x, cnn::real* v, cnn::real *r, cnn::real scale, cnn::real lambda, cnn::real momentum, cnn::real rho, cnn::real epsilon, cnn::real grd_squared_norm) {
    auto tb = SizeToBlockThreadPair(n);
    /// it may be more efficient to compute in cpu and not do reduce in gpu, but my observation is not 
    /// that case
    *r = rho * (*r) + (1 - rho) * grd_squared_norm;
    cnn::real den = sqrt(*r + epsilon);
    accTripletExprKernel << <tb.first, tb.second >> >(n, x, g, v, x, FL2SGDMomentumUpdate(lambda, scale / den, momentum));
    //CUDA_CHECK(cudaFree(sqnorm));
}

void sqeucdist(int n, const cnn::real* x0, const cnn::real *x1, cnn::real* y) {
  auto tb = SizeToBlockThreadPair(n);
  ker_sqeucdist<<<tb.first,tb.second>>>(n, x0, x1, y);
}

void l2_norm_reducer(int n, const cnn::real* x0, cnn::real* y, bool square, bool accumulate) {
  auto tb = SizeToBlockThreadPair(n);
  ker_l2_norm_reducer<<<tb.first,tb.second>>>(n, x0, y, square, accumulate);
}

void sqrt_of_l2_norm_reducer(int n, cnn::real* x0, cnn::real& res)
{
    thrust::device_ptr<cnn::real> dv_ptr = thrust::device_pointer_cast(x0);
    FSquare unary_op;
    thrust::plus<cnn::real> binary_op;
    res = std::sqrt(thrust::transform_reduce(dv_ptr, dv_ptr + n, unary_op, 0.0, binary_op));
}

void vector_sum(int rows, int cols, const cnn::real * a, cnn::real* c, const bool isColWise)
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
    _vector_sum<cnn::real> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(c, a, n, m, isColWise);
    cudaEventRecord(done);
    cudaEventSynchronize(done);
    cudaEventDestroy(done);
}

void vector_add_const(int rows, int cols, const cnn::real * a, int brow, int bcol, const cnn::real* b, cnn::real * c, bool isColWise)
{
    assert(rows > 0 && cols > 0); // converting from size_t to int may cause overflow

    int m = cols;
    int n = rows;

    if (brow != bcol && brow != 1)
        cuda_exception("const dimension has to be a scalar");

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
    _vector_add_const<cnn::real> << <blocksPerGrid, MAX_THREADS_PER_BLOCK, 0, cudaStreamDefault >> >(c, a, n, m, b, isColWise);
    cudaEventRecord(done);
    cudaEventSynchronize(done);
    cudaEventDestroy(done);
}

/// assume that a is a vector with col dimension
void row_element_multiply_with(int arow, int acol, const cnn::real * a, int brow, int bcol, cnn::real * b)
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
    _rowElementMultiplyWith<cnn::real> << <blocksPerGrid, MAX_THREADS_PER_BLOCK >> >(b, a, N, M);
    cudaEventRecord(done);
    cudaEventSynchronize(done);
    cudaEventDestroy(done);
}

void logsoftmax(int row, int col, const cnn::real* x0, cnn::real* y) 
{
    cudaStream_t t_stream = cudaStreamDefault;

    int N = col;
    int M = row;
    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);
    /// TO-DO: The N is the number of columns and is also the number of blocks. For small N, it is fine. For very large N, it may slow down computation. 
    _assignColumnwiseLogSoftmaxOf<cnn::real> << <N, 512, 0, t_stream >> >(x0, y, N, M);
    
    cudaEventRecord(done);
    
    cudaEventSynchronize(done);
    
    cudaEventDestroy(done);
}

void logsoftmax_backward(int row, int col, const cnn::real *fx, const cnn::real *dEdf, cnn::real *dEdx, cnn::real * gpu_softmax, cnn::real *grd)
{
    vexp(row * col, fx, gpu_softmax);
    vector_sum(row, col, dEdf, grd, true);
    row_element_multiply_with(1, col, grd, row, col, gpu_softmax);

    auto tb = SizeToBlockThreadPair(col * row);
    accBinaryExprKernel << <tb.first, tb.second >> >(col * row, dEdf, gpu_softmax, dEdx, FSubtract());
}

/** 
softmax opreations using cudnn
notice that cuNN uses rwo-major. 
so the N here is col. 
*/
void softmax(int row, int col, const cnn::real* x0, cnn::real* y)
{
    cudnnTensorDescriptor_t pInputDesc;
    int n = col; int c = 1; int h = 1; int w = row;

    cnn::real one = 1.0, zero = 0.0;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&pInputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pInputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, n, c, h, w));
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &one, pInputDesc, x0,
        &zero, pInputDesc, y));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(pInputDesc));
}

void softmax_backward(int row, int col, const cnn::real *fx, const cnn::real *dEdf, cnn::real *dEdx)
{
    cudnnTensorDescriptor_t pInputDesc;
    int n = col; int c = 1; int h = 1; int w = row; 
    cnn::real one = 1.0;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&pInputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pInputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, n, c, h, w));
    CHECK_CUDNN(cudnnSoftmaxBackward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &one, pInputDesc, fx, pInputDesc, dEdf,
        &one, pInputDesc, dEdx));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(pInputDesc));
}

/*
old implementation
void softmax(int row, int col, const cnn::real* x0, cnn::real* y)
{
    cudaStream_t t_stream = cudaStreamDefault;

    int N = col;
    int M = row;
    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);
    _assignColumnwiseSoftmaxOf<cnn::real> << <N, MAX_THREADS_PER_BLOCK, 0, t_stream >> >(x0, y, N, M);

    cudaEventRecord(done);
    cudaEventSynchronize(done);
    cudaEventDestroy(done);
}

///
/// see http://research.microsoft.com/pubs/226641/CNTKBook-20160217..pdf
/// input gradient += (\frac{\partial J}{\partial v_{ij}} - \sum_r \frac{\partial J}{\partial v_{rj} v_{rj}) v_{ij}

void softmax_backward(int row, int col, const cnn::real *fx, const cnn::real *dEdf, cnn::real *dEdx)
{
    cudaStream_t t_stream = cudaStreamDefault;
    cudaEvent_t done = nullptr;
    cudaEventCreate(&done);

    _assignColumnwiseSoftmaxOfBackward<cnn::real> << <col, MAX_THREADS_PER_BLOCK, 0, t_stream >> >(fx, dEdf, dEdx, col, row);

    cudaEventRecord(done);
    cudaEventSynchronize(done);
    cudaEventDestroy(done);
}
*/

// adapted from NVIDIA example
__global__ void ker_pnlsoftmax(int n, int elem_idx, const cnn::real *x0, cnn::real* res, cnn::real* logz) {
  __shared__ cnn::real buf[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      cnn::real me = __int_as_float(0xff800000);
      for (int pos = i; pos < n; pos += 256) {
      const cnn::real d = x0[pos];
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
  const cnn::real max_elem = buf[0];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    cnn::real sum = 0;
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
    cnn::real lz = log(buf[0]) + max_elem;
    logz[0] = lz;
    res[0] = lz - x0[elem_idx];
  }
}

void pnlsoftmax(int n, int elem_idx, const cnn::real* x0, cnn::real* y, cnn::real* logz) {
  auto tb = SizeToBlockThreadPair(n);
  ker_pnlsoftmax<<<tb.first,tb.second>>>(n, elem_idx, x0, y, logz);
}

__global__ void fixup_pnl(const cnn::real* dEdf, cnn::real* dEdxi, int i) {
  if (threadIdx.x == 0) dEdxi[i] -= dEdf[0];
}

void pnlsoftmax_backward(int n, int elem_idx, const cnn::real* x0, const cnn::real* dEdf, const cnn::real* logz, cnn::real* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  accUnaryExprKernel<<<tb.first, tb.second>>>(n, x0, dEdx, FPtrNegLogSoftmaxBackward(logz, dEdf));
  fixup_pnl<<<1,1>>>(dEdf, dEdx, elem_idx);
}


/**
conv1dnarrow using cuDNN, which is faster. however, cudnn is row-major as [n,c,h,w].
we always assume column-major, 
to accomodate to cudnn, n,c,h,w are interprated as 
[ncols, 1, nrows, 1]
can only do 1d convolution for each column

# CUDNN/Caffe sizes for various arrays in column-major notation:
conv x: (N,C,H,W): W,H=image size, C=channels, N=instances
conv w: (K,C,Y,X): X,Y=filter size, C=input channels, K=output channels
conv y: (N,K,H-Y+1,W-X+1)
conv b: (1,K,1,1)
*/
void conv2dnarrow(const cnn::real* kscalar_one, const cnn::real* kscalar_zero,
    const int xrow, const int xcol, const cnn::real* xs, 
    const int i_wkspace_sz, cnn::real* wkspace, 
    const int frow, const int fcol, const cnn::real *fx, 
    const int yrow, const int ycol, cnn::real *fy)
{
    cudnnTensorDescriptor_t pInputDesc;
    cudnnTensorDescriptor_t pOutputDesc;
    cudnnFilterDescriptor_t pFilterDesc = nullptr;
    cudnnConvolutionDescriptor_t pConvDesc = nullptr;
    int n = 1; int c = 1; int h = xcol; int w = xrow;
    int k_pFilter_in = 1; /// number of output feature maps
    int c_pFilter_in = 1; /// number of input feature maps
    int h_pFilter_in = fcol;
    int w_pFilter_in = frow;
    int n_out, c_out, h_out, w_out;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&pInputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&pOutputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&pFilterDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&pConvDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pInputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, n, c, h, w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(pFilterDesc, cudnnDataType, k_pFilter_in, c_pFilter_in, h_pFilter_in, w_pFilter_in));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(pConvDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    /// get the output layout
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(pConvDesc, pInputDesc, pFilterDesc, &n_out, &c_out, &h_out, &w_out));
    assert(n_out * c_out * h_out * w_out == yrow * ycol);

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pOutputDesc, CUDNN_TENSOR_NCHW, cudnnDataType, n_out, c_out, h_out, w_out));

    size_t sz_wkspace;
    bool   bNeedAllocateNewSpace = false;
    cnn::real *tmp_work_space;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, pInputDesc, pFilterDesc, pConvDesc, pOutputDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &sz_wkspace));
    if (sz_wkspace < i_wkspace_sz)
    {
        tmp_work_space = wkspace;
    }
    else{
        bNeedAllocateNewSpace = true;
        CUDA_CHECK(cudaMalloc(&tmp_work_space, sz_wkspace));
    }

    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle, kscalar_one, pInputDesc, xs, pFilterDesc, fx,
        pConvDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, tmp_work_space, sz_wkspace, kscalar_zero, pOutputDesc, fy));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(pInputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(pOutputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(pFilterDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(pConvDesc));

    if (bNeedAllocateNewSpace)
        CUDA_CHECK(cudaFree(tmp_work_space));
}

void conv1dwide(const int n, const int m, const cnn::real* xs, 
    const int k, const cnn::real *fx, cnn::real *fy)
{

    thrust::device_vector<cnn::real> dv((m + k) * n, 0.0);
    thrust::device_ptr<cnn::real> vp = dv.data();
    thrust::device_ptr<cnn::real> fp((cnn::real*)fx);
    thrust::device_ptr<cnn::real> xp((cnn::real*)xs);
    thrust::device_ptr<cnn::real> yp(fy);

    for (size_t tk = 0; tk < k; tk++)
    {
        for (size_t j = 0; j < m; j++)
            thrust::transform(xp + j * n, xp + (j + 1) * n, fp + tk * n, vp + tk * n + j * n, thrust::multiplies<cnn::real>());
    }
    thrust::copy(vp, vp + (m + k) * n, thrust::device_pointer_cast(fy));
}

void conv1dwide_backward(const int i, const int n, const int m, const cnn::real* xs, const int k, const cnn::real *fx, const cnn::real* dEdf, cnn::real *dEdx)
{
    thrust::device_vector<cnn::real> dv(m  * n, 0.0);
    thrust::device_ptr<cnn::real> vp = dv.data();
    thrust::device_ptr<cnn::real> fp((cnn::real*)fx);
    thrust::device_ptr<cnn::real> xp((cnn::real*)xs);
    thrust::device_ptr<cnn::real> d((cnn::real*)dEdf);
    thrust::device_ptr<cnn::real> yp(dEdx);

    for (size_t tk = 0; tk < k; tk++)
    {
        if (i == 0) { // derivative wrt input x
            for (size_t j = 0; j < m; j++)
                thrust::transform(d + j * n + tk*n, d + (j + 1) * n + tk*n, fp + tk * n, dv.data() + j * n, thrust::multiplies<cnn::real>());
        }
        else { // derivative wrt filter f
            for (size_t j = 0; j < m; j++)
                thrust::transform(d + j * n + tk*n, d + (j + 1) * n + tk*n, xp + j * n, dv.data() + tk * n, thrust::multiplies<cnn::real>());
        }
    }
    if (i == 0)
        thrust::transform(dv.data(), dv.data() + m * n, yp, yp, thrust::plus<cnn::real>());
    else 
        thrust::transform(dv.data(), dv.data() + k * n, yp, yp, thrust::plus<cnn::real>());
}

void addVectorToAllColumns(const int n, const cnn::real * xs, const int m, const cnn::real* fx, cnn::real *fy)
{
    thrust::device_ptr<cnn::real> fp((cnn::real*)fx);
    thrust::device_ptr<cnn::real> xp((cnn::real*)xs);
    thrust::device_ptr<cnn::real> yp(fy);
    for (size_t j = 0; j < n / m; j++)
        thrust::transform(xp + j * m, xp + (j + 1) * m, fp, yp + j * m, thrust::plus<cnn::real>());
}

void addVectorToAllColumns_backward(const int i, const int r, const int c, const cnn::real* dEdf, cnn::real *dEdxi)
{
    thrust::device_ptr<const cnn::real> dp(dEdf);
    thrust::device_ptr<cnn::real> dx(dEdxi);

    if (i == 0)
    {
        // x
        thrust::transform(dp, dp + r * c, dx, dx, thrust::plus<cnn::real>());
    }
    else
    {
        // bias
        for (int k = 0; k < c; k++)
            thrust::transform(dp + k * r, dp + (k + 1)*r, dx, dx, thrust::plus<cnn::real>());
    }
}

/**
stride : the jump step
*/
void foldRows(const int n, const int m, const cnn::real *xs, const int stride, const int orows, cnn::real *fy)
{
    thrust::device_ptr<cnn::real> xp((cnn::real*)xs), pp;
    thrust::device_ptr<cnn::real> yp(fy);
    thrust::host_vector<cnn::real> vo(orows * m);

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

void foldRows_backward(const int orows, const cnn::real* dEdf, const int n, const int m, cnn::real *fy)
{
    thrust::device_ptr<cnn::real> dp((cnn::real*)dEdf);
    thrust::device_ptr<cnn::real> yp(fy);

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

void kMaxPooling(const int n, const int m, const cnn::real *xs, const int k, cnn::real *fy, int* aux_mem)
{
    thrust::device_ptr<cnn::real> xp((cnn::real*)xs), pp;
    thrust::device_ptr<cnn::real> yp(fy);
    thrust::device_vector<cnn::real> vo(m);
    thrust::device_vector<cnn::real> vp(k);

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

void kMaxPooling_backward(const int n, const int m, const cnn::real *xs, const int k, const cnn::real * dEdf, cnn::real *dEdxi, const int* aux_mem)
{
    const int* maxmap = aux_mem;
    int mk = 0;
    int oj;
    thrust::device_ptr<const cnn::real> xp(xs);
    thrust::device_ptr<const cnn::real> dp(dEdf);
    thrust::device_ptr<cnn::real> yp(dEdxi);
    thrust::host_vector<int> hv(n, 0);
    cudaMemcpy(hv.data(), maxmap, sizeof(int)*n, cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = 0; j < k; ++j) {
            oj = hv[mk++];
            if (oj < k && oj >= 0){
                thrust::transform(dp + i + j * n, dp + i + j * n + 1, yp + i + oj * n, yp + i + oj * n, thrust::plus<cnn::real>());
            }
        }
    }
}


} // namespace gpu
} // namespace cnn
