#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#include "cnn/gpu-kernels.h"
#include "cnn/functors.h"
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace cnn {
namespace gpu {

// this wraps kernel dispatches for various operations (preventing us from
// having to compile a version of nodes.cc with NVCC)

struct scale_functor
{
    const float a;

    scale_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x) const
    {
        return a * x;
    }
};

struct saxpy_functor
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        float operator()(const float& x, const float& y) const
    {
        return a * x + y;
    }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void set_to_value_of(int n, float* x0, float val) {
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(x0);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + n, val);
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
  unaryExprKernel<<<tb.first, tb.second>>>(n, x, y, FConstantMinus(c));
}

void vexp(int n, const float* x, float* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FExp());
}

void vlog(int n, const float* x, float* y) {
    auto tb = SizeToBlockThreadPair(n);
    unaryExprKernel << <tb.first, tb.second >> >(n, x, y, FLog());
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
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, x, g, x, FL2SGDUpdate(lambda, scale));
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

void sqeucdist(int n, const float* x0, const float *x1, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  ker_sqeucdist<<<tb.first,tb.second>>>(n, x0, x1, y);
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

void l2_norm_reducer(int n, const float* x0, float* y, bool square, bool accumulate) {
  auto tb = SizeToBlockThreadPair(n);
  ker_l2_norm_reducer<<<tb.first,tb.second>>>(n, x0, y, square, accumulate);
}

// adapted from NVIDIA example
__global__ void ker_softmax(int n, const float *x0, float* res) {
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
  float lz = log(buf[0]) + max_elem;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    res[i] = exp(x0[i] - lz);
    i += gridDim.x * blockDim.x;
  }
}

void softmax(int n, const float* x0, float* y) {
  auto tb = SizeToBlockThreadPair(n);
  ker_softmax<<<tb.first,tb.second>>>(n, x0, y);
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

void softmax_backward(int n, const float* fx, const float* dEdf, float* dEdx) {
  auto tb = SizeToBlockThreadPair(n);
  float* gpu_ods;
  float ods;
  cudaMalloc((void **)&gpu_ods, sizeof(float));
  ker_dotproduct<<<tb.first, tb.second>>>(n, fx, dEdf, gpu_ods);
  cudaMemcpy(&ods, gpu_ods, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(gpu_ods);
  accBinaryExprKernel<<<tb.first, tb.second>>>(n, fx, dEdf, dEdx, FSoftmaxBackward(-ods));
}

void logsoftmax_backward(int n, const float* fx, const float* dEdf, float* dEdx) 
{
    /*
    float off_diag_sum = -(*fx).binaryExpr(*dEdf, FWeightedError()).sum();
    *dEdxi += (*fx).binaryExpr(*dEdf, FLogSoftmaxBackward(off_diag_sum));
    */
    thrust::device_ptr<float> dp = thrust::device_pointer_cast((float*)fx);
    thrust::device_ptr<float> de = thrust::device_pointer_cast((float*)dEdf);
    thrust::device_ptr<float> dr = thrust::device_pointer_cast(dEdx);
    thrust::device_vector<float> dtemp(n);
    thrust::transform(dp, dp + n, de, dtemp.begin(), FWeightedError());
    float off_diag_sum  = - thrust::reduce(dtemp.begin(), dtemp.end());
    thrust::transform(dp, dp + n, de, dtemp.begin(), FLogSoftmaxBackward(off_diag_sum)); 
    thrust::transform(dtemp.begin(), dtemp.end(), dr, dr, thrust::plus<float>());
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

} // namespace gpu
} // namespace cnn
