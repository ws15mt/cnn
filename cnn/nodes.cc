#include "cnn/nodes.h"

#include <limits>
#include <cmath>
#include <stdexcept>
#include "cnn/macros.h"
#include "cnn/simd-functors.h"
#include "cnn/functors.h"
#if HAVE_CUDA
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
using namespace cnn::gpu;
#endif


using namespace std;


// notes on implementing differentiable components
// 1) fx can be understood as a pointer to the (preallocated) location for the result
//    of forward to be stored
// 2) fx is not initialized, so after calling forward fx must point to the correct answer
// 3) fx can be repointed to an input, if forward(x) evaluates to x (e.g., in reshaping)
// 4) dEdxi MUST **ACCUMULATE** a result since multiple calls to forward may depend on
//    the same x_i. Even, e.g., Identity must be implemented as
//    dEdx1 += dEdf. THIS IS EXTREMELY IMPORTANT
// 5) scalars results of forward are placed in fx.v[0]
// 6) CNN manages its own memory, not Eigen, and it is configured with the
//    EIGEN_NO_MALLOC option. If you get an error about Eigen attempting to allocate
//    memory, it is (probably) because of an implicit creation of a temporary variable.
//    To tell Eigen this is not necessary, the noalias() method is available. If you really
//    do need a temporary variable, its capacity must be requested by Node::aux_storage_space
//
// notes on debugging problems with differentiable components
// 1) fx is uninitialized when forward is called- are you relying on it being 0?
// 2) dEdxi must accummulate (see point 4 above!)
//

namespace cnn {

void Pow::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Pow not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = xs[1]->v[0];
  (*fx).array() = x1.array().pow(x2);
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void Pow::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Pow not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = xs[1]->v[0];
  if (i == 0) {
    *dEdxi += (x2 * x1.array().pow(x2 - 1).matrix()).cwiseProduct(*dEdf);
  } else {
    // y = a^x
    // dy/dx = a^x * log(a)
    (*dEdxi).noalias() += (*fx).cwiseProduct(x1.array().log().matrix()).transpose() * (*dEdf);
  }
#endif
}

size_t Min::aux_storage_size() const {
  return dim.size() * sizeof(cnn::real);
}

void Min::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Min not yet implemented for CUDA");
#else
  auto y = *fx;
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  Tensor t(fx.d, static_cast<cnn::real*>(aux_mem), xs[0]->m_device_id);
  auto u = *t;
  u = (x1.array() < x2.array()).matrix().cast<cnn::real>();
  y = x1.cwiseMin(x2);
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void Min::backward_impl(const vector<const Tensor*>& xs,
                   const Tensor& fx,
                   const Tensor& dEdf,
                   unsigned i,
                   Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Min not yet implemented for CUDA");
#else
  const Tensor t(dEdxi.d, static_cast<cnn::real*>(aux_mem), xs[0]->m_device_id);
  if (i == 0) {
    *dEdxi += (*t).cwiseProduct(*dEdf);
  } else {
    *dEdxi += (*t).binaryExpr(*dEdf, FMaxBackwardInv());
  }
#endif
}

size_t Max::aux_storage_size() const {
  return dim.size() * sizeof(cnn::real);
}

void Max::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Max not yet implemented for CUDA");
#else
  auto y = *fx;
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  Tensor t(fx.d, static_cast<cnn::real*>(aux_mem), xs[0]->m_device_id);
  auto u = *t;
  u = (x1.array() > x2.array()).matrix().cast<cnn::real>();
  y = x1.cwiseMax(x2);
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void Max::backward_impl(const vector<const Tensor*>& xs,
                   const Tensor& fx,
                   const Tensor& dEdf,
                   unsigned i,
                   Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("Max not yet implemented for CUDA");
#else
  const Tensor t(dEdxi.d, static_cast<cnn::real*>(aux_mem), xs[0]->m_device_id);
  if (i == 0) {
    *dEdxi += (*t).cwiseProduct(*dEdf);
  } else {
    *dEdxi += (*t).binaryExpr(*dEdf, FMaxBackwardInv());
  }
#endif
}

void TraceOfProduct::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("TraceOfProduct not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 * x2.transpose()).trace();
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void TraceOfProduct::backward_impl(const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("TraceOfProduct not yet implemented for CUDA");
#else
  const cnn::real d = dEdf.v[0];
  auto xother = **xs[1 - i];
  *dEdxi += d * xother;
#endif
}

void ConstScalarMultiply::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
    gpu::vconstant_multiplyx(fx.d.size(), alpha, xs[0]->v, fx.v);
#else
    *fx = (**xs[0]) * alpha;
#endif
    fx.m_device_id = xs[0]->m_device_id;
}

void ConstScalarMultiply::backward_impl(const vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const 
{
    assert(i == 0);
#if HAVE_CUDA
    gpu::vconstant_multiplyx_backward(fx.d.size(), alpha, dEdf.v, dEdxi.v);
#else
    *dEdxi += *dEdf * alpha;
#endif
}

void DotProduct::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("DotProduct not yet implemented for CUDA");
#else
    *fx = (**xs[0]).transpose() * (**xs[1]);
#endif
    fx.m_device_id = xs[0]->m_device_id;
}

void DotProduct::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("DotProduct not yet implemented for CUDA");
#else
  (*dEdxi) += (dEdf.v[0]) * (**xs[1 - i]);
#endif
}

void Transpose::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (dim.rows() == 1 || dim.cols() == 1) {
#if HAVE_CUDA
      CUDA_CHECK(cudaMemcpy(fx.v, xs[0]->v, sizeof(cnn::real) * dim.size(), cudaMemcpyDeviceToDevice));
#else
      fx.v = xs[0]->v;
#endif
  }
  else {
#ifdef HAVE_CUDA
#ifdef USE_DOUBLE
      CUBLAS_CHECK(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, fx.d.rows(), fx.d.cols(),
          kSCALAR_ONE, xs[0]->v, xs[0]->d.rows(), kSCALAR_ZERO, xs[0]->v, xs[0]->d.rows(), fx.v, fx.d.rows()));
#else
      CUBLAS_CHECK(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, fx.d.rows(), fx.d.cols(), 
              kSCALAR_ONE, xs[0]->v, xs[0]->d.rows(), kSCALAR_ZERO, xs[0]->v, xs[0]->d.rows(), fx.v, fx.d.rows()));
#endif
#else
    for(unsigned b = 0; b < xs[0]->d.bd; ++b)
      fx.batch_matrix(b).noalias() = xs[0]->batch_matrix(b).transpose();
#endif
  }
  fx.m_device_id = xs[0]->m_device_id;
}

void Transpose::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#if HAVE_CUDA
  /// for usage of of cublassegeam see 
  /// http://scikit-cuda.readthedocs.org/en/latest/generated/skcuda.cublas.cublasSgeam.html
#ifdef USE_DOUBLE
    CUBLAS_CHECK(cublasDgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, dEdxi.d.rows(), dEdxi.d.cols(),
            kSCALAR_ONE, dEdf.v, dEdf.d.rows(), kSCALAR_ONE, dEdxi.v, dEdxi.d.rows(), dEdxi.v, dEdxi.d.rows()));
#else
    CUBLAS_CHECK(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, dEdxi.d.rows(), dEdxi.d.cols(),
            kSCALAR_ONE, dEdf.v, dEdf.d.rows(), kSCALAR_ONE, dEdxi.v, dEdxi.d.rows(), dEdxi.v, dEdxi.d.rows()));
#endif
#else
  for(unsigned b = 0; b < xs[0]->d.bd; ++b)
    dEdxi.batch_matrix(b) += dEdf.batch_matrix(b).transpose();
#endif
}

void Reshape::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  // BUG will be created if just point to the input memory and change dimensions
#if HAVE_CUDA
  CUDA_CHECK(cudaMemcpy(fx.v, xs[0]->v, sizeof(cnn::real) * dim.size(), cudaMemcpyDeviceToDevice));
#else
  fx.v = xs[0]->v;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void Reshape::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  const Tensor reshaped(dEdxi.d, dEdf.v, device_id);
#ifdef HAVE_CUDA
  gpu::vconstant_multiplyx_backward(reshaped.d.size(), 1.0, reshaped.v, dEdxi.v); 
#else
  *dEdxi += *reshaped;
#endif
}

void SumColumns::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  auto y = *fx;
  if (xs.size() == 1) {
    y = x.rowwise().sum();
  } else {
    throw std::invalid_argument("two inputs in SumColumns::forward!");
  }
  fx.m_device_id = xs[0]->m_device_id;
}

void SumColumns::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  auto out = *dEdxi;
  // this uses Eigen's broadcast capability
  // the following doesn't compile, so i use the next line
  //out.colwise() += *dEdf;
  out.colwise() += (*dEdf).col(0);
}

void KMHNGram::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto x = **xs[0];
  const int new_cols = x.cols() - n + 1;
  assert(new_cols > 0);
  auto res = *fx;
  res.setZero();
  for (int j = 0; j < new_cols; ++j) {
    auto c_j = res.col(j);
    for (unsigned k = 0; k < n; ++k)
      c_j += x.col(j + k);
  }
  fx.m_device_id = xs[0]->m_device_id;
}

void KMHNGram::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  const int c = dEdf.d.cols();
  for (int j = 0; j < c; ++j)
    for (unsigned k = 0; k < n; ++k)
      (*dEdxi).col(j+k) += (*dEdf).col(j);
}

//   Y_ij = A_ijk * B_k (+ C_ij)
void InnerProduct3D_1D::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  auto b = **xs[1];
  auto y = *fx;
  const unsigned i = y.rows();
  const unsigned j = y.cols();
  const unsigned k = b.rows();
  // the following reshape tensors into order 1 or 2 sizes
  // but they point to the same memory
  Tensor ta({i*j,k}, xs[0]->v, device_id);
  Tensor ty({i*j}, fx.v, device_id);
  auto A = *ta;
  if (xs.size() == 3) {
      Tensor tc({ i*j }, xs[2]->v, device_id);
    auto c = *tc;
    // want to do A * b + c, but it triggers memory allocation
    (*ty) = c;
    (*ty).noalias() += A * b;
  } else {
    assert(xs.size() == 2);
    (*ty).noalias() = A * b;
  }
  fx.m_device_id = xs[0]->m_device_id;
}

void InnerProduct3D_1D::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  auto b = **xs[1];
  auto y = *fx;
  const unsigned si = y.rows();
  const unsigned sj = y.cols();
  const unsigned sk = b.rows();
  Tensor tdEdf({ si*sj }, dEdf.v, device_id);
  if (i == 0) { // 3-tensor
      Tensor tdEdxi({ si*sj, sk }, dEdxi.v, device_id);
    (*tdEdxi).noalias() += *tdEdf * (**xs[1]).transpose();
  } else if (i == 1) { // vector
      Tensor ta({ si*sj, sk }, xs[0]->v, device_id);
    (*dEdxi).noalias() += (*ta).transpose() * *tdEdf;
  } else { // matrix bias
    *dEdxi += *dEdf;
  }
}

size_t GaussianNoise::aux_storage_size() const {
  return dim.size() * sizeof(cnn::real);
}

void GaussianNoise::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("GaussianNoise not yet implemented for CUDA");
#else
    Tensor m(dim, (cnn::real*)aux_mem, xs[0]->m_device_id);
  TensorTools::RandomizeNormal(0, stddev, m);
  (*fx) = **xs[0] + *m;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void GaussianNoise::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("GaussianNoise not yet implemented for CUDA");
#else
  *dEdxi += *dEdf;
#endif
}

size_t Dropout::aux_storage_size() const {
  return dim.size() * sizeof(cnn::real);
}

void Dropout::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Dropout not yet implemented for CUDA");
#else
    Tensor m(dim, (cnn::real*)aux_mem, xs[0]->m_device_id);
  TensorTools::RandomBernoulli(m, (1.f-p), 1.f / (1.f-p));
  fx.vec() = xs[0]->vec().cwiseProduct(m.vec());
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void Dropout::backward_impl(const vector<const Tensor*>& xs,
                       const Tensor& fx,
                       const Tensor& dEdf,
                       unsigned i,
                       Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Pow not yet implemented for CUDA");
#else
    Tensor m(dim, (cnn::real*)aux_mem, xs[0]->m_device_id);
  dEdxi.vec() += dEdf.vec().cwiseProduct(m.vec());
#endif
}

size_t BlockDropout::aux_storage_size() const {
  // we just need to remember whether this entire block is turned on (1.0) or off (0.0)
  return 1 * sizeof(cnn::real);
}

void BlockDropout::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BlockDropout not yet implemented for CUDA");
#else
  bernoulli_distribution distribution(1.0 - dropout_probability);
  cnn::real block_multiplier = distribution(*rndeng)? 1.0 : 0.0;
  block_multiplier = 
    dropout_probability == 1.0? 0.0 : block_multiplier / (1.0 - dropout_probability);
  if (dropout_probability > 1.0 || dropout_probability < 0.0) {
    assert(false && "dropout probability must be in the range [0, 1]");
  }
  *(static_cast<cnn::real*>(aux_mem)) = block_multiplier;
  (*fx) = **xs[0] * block_multiplier;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void BlockDropout::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BlockDropout not yet implemented for CUDA");
#else
  cnn::real block_multiplier = *(static_cast<cnn::real*>(aux_mem));
  (*dEdxi) += (*dEdf) * block_multiplier;
#endif
}

void ConstantPlusX::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("ConstantPlusX not yet implemented for CUDA");
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(const_add_op<cnn::real>(c));
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void ConstantPlusX::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  *dEdxi += *dEdf;
}

void ConstantMinusX::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vconstant_minusx(fx.d.size(), c, xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(const_minus_op<cnn::real>(c));
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void ConstantMinusX::backward_impl(const vector<const Tensor*>& xs,
                              const Tensor& fx,
                              const Tensor& dEdf,
                              unsigned i,
                              Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vnegate_backward(dEdxi.d.size(), dEdf.v, dEdxi.v);
#else
  *dEdxi -= *dEdf;
#endif
}

template <class T>
EIGEN_STRONG_INLINE cnn::real logsumexp(const T& x) {
  using std::exp;
  using std::log;
  const cnn::real m = x.maxCoeff();
#if 1
  // these are equivalent, but this can use vectorized arithmetic
  cnn::real z = x.unaryExpr(const_add_op<cnn::real>(-m)).array().exp().matrix().sum();
#else
  cnn::real z = 0;
  for (unsigned i = 0; i < x.rows(); ++i)
    z += exp(x(i,0) - m);
#endif
  return m + log(z);
}

// this i need to do something better, but this is a work-around
// if this is too small, just make it bigger
#define MAX_LOG_SUM_EXP 65536
size_t LogSumExp::aux_storage_size() const {
  return MAX_LOG_SUM_EXP * sizeof(cnn::real);
}

void LogSumExp::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  if (num_args == 1) {
    fx.v = xs[0]->v;
    return;
  }
  for (unsigned i = 0; i < xs.size(); ++i)
    static_cast<cnn::real*>(aux_mem)[i] = (**xs[i])(0,0);
  Dim r = {(unsigned int)xs.size()};
  Tensor v(r, static_cast<cnn::real*>(aux_mem), device_id);
  fx.v[0] = logsumexp(*v);
  fx.m_device_id = xs[0]->m_device_id;
}

void LogSumExp::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  if (xs.size() == 0) {
    *dEdxi += *dEdf;
    return;
  }
  // df/dx_i = 1/{sum_j exp(x_j)} * exp(x_i)}
  //         = 1/{exp f(x)} * exp(x_i)
  //         = exp(x_i - f(x))
  auto d = *dEdxi;
  d.array() += (**xs[i] - *fx).array().exp() * (*dEdf).array();
}

void Reduce::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    const unsigned num_args = xs.size();
    assert(num_args == 1 && fx.d.cols() == 1);
#if HAVE_CUDA
    l2_norm_reducer(xs[0]->d.size(), xs[0]->v, fx.v, false, false);
#else
    fx.v[0] = 0.0; 
    for (int k = 0; k < xs[0]->d.size(); k++)
        fx.v[0] += xs[0]->v[k]; 
#endif
    fx.m_device_id = xs[0]->m_device_id;
    return;
}

void Reduce::backward_impl(const vector<const Tensor*>& xs,
    const Tensor& fx,
    const Tensor& dEdf,
    unsigned i,
    Tensor& dEdxi) const {
#if HAVE_CUDA
    l2_norm_reducer(1, dEdf.v, dEdxi.v, false, true);
#else
    for (int k = 0; k < dEdxi.d.size(); k++)
        dEdxi.v[k] += dEdf.v[0];
#endif
}

void Sum::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    const unsigned num_args = xs.size();
    fx.m_device_id = xs[0]->m_device_id;
    if (num_args == 1) {
#if HAVE_CUDA
        CUDA_CHECK(cudaMemcpy(fx.v, xs[0]->v, sizeof(cnn::real) * dim.size(), cudaMemcpyDeviceToDevice));
#else
        fx.v = xs[0]->v;
#endif
        return;
    }
#if HAVE_CUDA
    TensorTools::Zero(fx);
    for (unsigned i = 0; i < num_args; ++i)
    {
        if (sizeof(cnn::real) == sizeof(float))
        {
            CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(xs[i]->v), 1, reinterpret_cast<float*>(fx.v), 1));
        }
        else if (sizeof(cnn::real) == sizeof(double))
        {
            CUBLAS_CHECK(cublasDaxpy(cublas_handle, fx.d.size(), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(xs[i]->v), 1, reinterpret_cast<double*>(fx.v), 1));
        }
    }
#else
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = **xs[0]; break;
    case 2: res = **xs[0] + **xs[1]; break;
    case 3: res = **xs[0] + **xs[1] + **xs[2]; break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += **xs[i] + **xs[i+1] + **xs[i+2] + **xs[i+3];
#endif
}

void Sum::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {

#if HAVE_CUDA
    if (sizeof(cnn::real) == sizeof(float))
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(dEdf.v), 1, reinterpret_cast<float*>(dEdxi.v), 1));
    else if (sizeof(cnn::real) == sizeof(double))
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, fx.d.size(), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(dEdf.v), 1, reinterpret_cast<double*>(dEdxi.v), 1));
#else
  *dEdxi += *dEdf;
#endif
}

void SumBatches::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  unsigned num_args = xs[0]->d.bd;
  fx.m_device_id = xs[0]->m_device_id;
#if HAVE_CUDA
  TensorTools::Zero(fx);
  for (unsigned i = 0; i < num_args; ++i)
  {
      if (sizeof(cnn::real) == sizeof(float))
          CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(xs[0]->v + i * xs[0]->d.batch_size()), 1, reinterpret_cast<float*>(fx.v), 1));
      else if (sizeof(cnn::real) == sizeof(double))
          CUBLAS_CHECK(cublasDaxpy(cublas_handle, fx.d.size(), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(xs[0]->v + i * xs[0]->d.batch_size()), 1, reinterpret_cast<double*>(fx.v), 1));
  }
#else
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = xs[0]->batch_matrix(0); break;
    case 2: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1); break;
    case 3: res = xs[0]->batch_matrix(0) + xs[0]->batch_matrix(1) + xs[0]->batch_matrix(2); break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += xs[0]->batch_matrix(i) + xs[0]->batch_matrix(i+1) + xs[0]->batch_matrix(i+2) + xs[0]->batch_matrix(i+3);
#endif
}

void SumBatches::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
  assert(i == 0);
#if HAVE_CUDA
  for (unsigned i = 0; i < dEdxi.d.bd; ++i)
  {
      if (sizeof(cnn::real) == sizeof(float))
          CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(dEdf.v), 1, reinterpret_cast<float*>(dEdxi.v) + i * dEdxi.d.batch_size(), 1));
      else if (sizeof(cnn::real) == sizeof(double))
          CUBLAS_CHECK(cublasDaxpy(cublas_handle, fx.d.size(), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(dEdf.v), 1, reinterpret_cast<double*>(dEdxi.v) + i * dEdxi.d.batch_size(), 1));
  }
#else
  for (unsigned i = 0; i < dEdxi.d.bd; ++i)
    dEdxi.batch_matrix(i) += *dEdf;
#endif
}

size_t Average::aux_storage_size() const {
    return sizeof(cnn::real);
}

void Average::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  const unsigned num_args = xs.size();
  fx.m_device_id = xs[0]->m_device_id;
#if HAVE_CUDA
  TensorTools::Zero(fx);
  for (unsigned i = 0; i < num_args; ++i)
  {
      if (sizeof(cnn::real) == sizeof(float))
          CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(xs[i]->v), 1, reinterpret_cast<float*>(fx.v), 1));
      else if (sizeof(cnn::real) == sizeof(double))
          CUBLAS_CHECK(cublasDaxpy(cublas_handle, fx.d.size(), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(xs[i]->v), 1, reinterpret_cast<double*>(fx.v), 1));
  }
  gpu::vconstant_multiplyx(fx.d.size(), 1. / num_args, fx.v, fx.v);

#else
  auto res = *fx;
  const unsigned remainder = num_args % 4;
  switch (remainder) {
    case 0: res.setZero(); break;
    case 1: res = **xs[0]; break;
    case 2: res = **xs[0] + **xs[1]; break;
    case 3: res = **xs[0] + **xs[1] + **xs[2]; break;
  }
  for (unsigned i = remainder; i < num_args; i += 4)
    res += **xs[i] + **xs[i+1] + **xs[i+2] + **xs[i+3];
  res /= num_args;
#endif
}

/**
check if a 1/sz is allready computed and stored in GPU memory, 
if so, use it. otherwise, need to compute it and put it in the back_off_mem
*/
#ifdef HAVE_CUDA
cnn::real* pointer_to_one_over_size(int sz, cnn::real * back_off_mem)
{
    cnn::real *fdevptr = nullptr;
    if (sz > MEM_PRE_ALLOCATED_CONSTS_NUMBERS + 1)
    {
        cnn::real nbr_samples = 1. / sz;
        fdevptr = static_cast<cnn::real*>(back_off_mem);
        CUDA_CHECK(cudaMemcpy(fdevptr, &nbr_samples, sizeof(cnn::real), cudaMemcpyHostToDevice));
    }
    else if (sz == 1)
        fdevptr = kSCALAR_ONE;
    else{
        fdevptr = kSCALAR_ONE_OVER_INT[sz - 2];
    }
    return fdevptr;
}
#endif

void Average::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#if HAVE_CUDA
    /// to speed-up, use pre-allocated const, instead of doing cpu to gpu memcpy of the const
    cnn::real *fdevptr = pointer_to_one_over_size(xs.size(), static_cast<cnn::real*>(aux_mem));

    if (sizeof(cnn::real) == sizeof(float))
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), reinterpret_cast<float*>(fdevptr), reinterpret_cast<float*>(dEdf.v), 1, reinterpret_cast<float*>(dEdxi.v), 1));
    else if (sizeof(cnn::real) == sizeof(double))
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, fx.d.size(), reinterpret_cast<double*>(fdevptr), reinterpret_cast<double*>(dEdf.v), 1, reinterpret_cast<double*>(dEdxi.v), 1));
#else
    *dEdxi += (*dEdf / xs.size());
#endif
};

void Sqrt::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    fx.m_device_id = xs[0]->m_device_id;
#ifdef HAVE_CUDA
  throw std::runtime_error("Sqrt not yet implemented for CUDA");
#else
  auto x = **xs[0];
  (*fx) = x.cwiseSqrt();
#endif
}

void Sqrt::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Sqrt not yet implemented for CUDA");
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, FSqrtBackward());
#endif
}

/*void Erf::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Erf not yet implemented for CUDA");
#else
  auto x = **xs[0];
  (*fx).array() = x.array().erf();
#endif
}

void Erf::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Erf not yet implemented for CUDA");
#else
  auto x = **xs[0];
  *dEdxi += x.binaryExpr(*dEdf, scalar_erf_backward_op<cnn::real>());
#endif
}
*/

void Tanh::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    fx.m_device_id = xs[0]->m_device_id;
#if HAVE_CUDA
  gpu::vtanh(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  (*fx).array() = x.array().tanh();
#endif
}

void Tanh::backward_impl(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i,
                      Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vtanh_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, scalar_tanh_backward_op<cnn::real>());
#endif
}

void Square::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    fx.m_device_id = xs[0]->m_device_id;
#ifdef HAVE_CUDA
  throw std::runtime_error("Square not yet implemented for CUDA");
#else
  auto x = **xs[0];
  (*fx).array() = x.array().square();
#endif
}

void Square::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Square not yet implemented for CUDA");
#else
  auto x = **xs[0];
  *dEdxi += (*dEdf).cwiseProduct(x) * 2;
#endif
}

void Cube::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    fx.m_device_id = xs[0]->m_device_id;
#ifdef HAVE_CUDA
  throw std::runtime_error("Square not yet implemented for CUDA");
#else
  auto x = **xs[0];
  (*fx).array() = x.array().cube();
#endif
}

void Cube::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Cube not yet implemented for CUDA");
#else
  auto x = **xs[0];
//  *dEdxi += (*dEdf).cwiseProduct(x.cwiseProduct(x)) * 3;
  (*dEdxi).array() += (*dEdf).array() * x.array().square() * 3;
#endif
}

void Exp::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    fx.m_device_id = xs[0]->m_device_id;
#if HAVE_CUDA
    gpu::vexp(xs[0]->d.size(), xs[0]->v, fx.v); 
#else
    auto x = **xs[0];
    *fx = x.array().exp();
#endif
}

void Exp::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#if HAVE_CUDA
    gpu::vcwise_product_backward(xs[0]->d.size(), fx.v, dEdf.v, dEdxi.v);
#else
    *dEdxi += (*dEdf).cwiseProduct(*fx);
#endif
}

/* eigen speciffunctions.h is not working 
   TO-DO: need to figure out why
void LogGamma::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("LogGamma not yet implemented for CUDA");
#else
  auto x = **xs[0];
  *fx = x.array().lgamma();
#endif
}

void LogGamma::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("LogGamma not yet implemented for CUDA");
#else
  auto x = **xs[0];
  *dEdxi += x.binaryExpr(*dEdf, FLogGammaBackward());
#endif
}
*/

void Log::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    fx.m_device_id = xs[0]->m_device_id;
#if HAVE_CUDA
  gpu::vlog(fx.d.size(), xs[0]->v, fx.v);
#else
    auto x = **xs[0];
    *fx = x.array().log();
#endif
}

void Log::backward_impl(const vector<const Tensor*>& xs,
                     const Tensor& fx,
                     const Tensor& dEdf,
                     unsigned i,
                     Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vlog_backward(fx.d.size(), xs[0]->v, dEdf.v, dEdxi.v);
#else
  auto x = **xs[0];
  *dEdxi += (*dEdf).cwiseQuotient(x);
#endif
}

void Concatenate::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned rows = 0;
  fx.m_device_id = xs[0]->m_device_id;
  for (auto x : xs) rows += x->d.rows();
  // the following should use auxiliary memory
  src_row_indices.resize(xs.size());
  unsigned ind = 0;
  unsigned cols = xs[0]->d.cols();
  unsigned total_rows = 0;
  for (auto x : xs) {
      if (x->d.cols() != cols)
      {
          cerr << "columns need to the same for Concatenate " << endl;
          abort();
      }
      total_rows += x->d.rows();
  }
  
  unsigned k = 0;
  for (auto x : xs) {
    src_row_indices[k++] = ind;
    auto & xi = *x;
    const unsigned rows = xi.d.rows();
#if HAVE_CUDA
    /// relaxed to support multiple columns!
    int stt = ind;
    for (int k = 0; k < cols; k++) /// not efficient, unfortunately
    {
        CUDA_CHECK(cudaMemcpyAsync(&fx.v[stt], &xi.v[k * rows], sizeof(cnn::real) * rows, cudaMemcpyDeviceToDevice));
        stt += total_rows;
    }
#else
    (*fx).middleRows(ind, rows) = *xi;
#endif
    ind += rows;
  }
}

void Concatenate::backward_impl(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < src_row_indices.size());
  const unsigned total_rows = dEdf.d.rows();
  const unsigned cols = dEdxi.d.cols();
  const unsigned rows = dEdxi.d.rows();
  const unsigned begin = src_row_indices[i];
#if HAVE_CUDA
  /// not efficient unfortunately
  for (size_t k = 0; k < cols; k++)
  {
      if (sizeof(cnn::real) == sizeof(float))
          CUBLAS_CHECK(cublasSaxpy(cublas_handle, rows, reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(&dEdf.v[begin + k * total_rows]), 1, reinterpret_cast<float*>(&dEdxi.v[k * rows]), 1));
      if (sizeof(cnn::real) == sizeof(double))
          CUBLAS_CHECK(cublasDaxpy(cublas_handle, rows, reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(&dEdf.v[begin + k * total_rows]), 1, reinterpret_cast<double*>(&dEdxi.v[k * rows]), 1));
  }
#else
  *dEdxi += (*dEdf).middleRows(begin, rows);
#endif
}

void ConcatenateColumns::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  unsigned c = 0;

  fx.m_device_id = xs[0]->m_device_id;
  acc_col_size.clear();
  for (unsigned i = 0; i < xs.size(); ++i) {
    auto & xi = *xs[i];
    const unsigned cols = xi.d.cols();
#if HAVE_CUDA
    // CUBLAS matricies are column-major, so just copy the memory
    const unsigned rows = xi.d.rows();
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[rows*c], xi.v, sizeof(cnn::real) * rows * cols, cudaMemcpyDeviceToDevice));
#else
    (*fx).middleCols(c, cols) = **xs[i];
#endif
    c += cols;
    acc_col_size.push_back(c);
  }
}

void ConcatenateColumns::backward_impl(const vector<const Tensor*>& xs,
                                    const Tensor& fx,
                                    const Tensor& dEdf,
                                    unsigned i,
                                    Tensor& dEdxi) const {
  unsigned* pp = static_cast<unsigned*>(aux_mem);
  unsigned c = acc_col_size[i];
#if HAVE_CUDA
  const unsigned rows = dEdxi.d.rows();
  const unsigned cols = dEdxi.d.cols();
  const unsigned begin = (c - cols)*rows;
  if (sizeof(cnn::real) == sizeof(float))
      CUBLAS_CHECK(cublasSaxpy(cublas_handle, rows * cols, reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(&dEdf.v[begin]), 1, reinterpret_cast<float*>(dEdxi.v), 1));
  else if (sizeof(cnn::real) == sizeof(double))
      CUBLAS_CHECK(cublasDaxpy(cublas_handle, rows * cols, reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(&dEdf.v[begin]), 1, reinterpret_cast<double*>(dEdxi.v), 1));
#else
  auto dEdx = *dEdxi;
  int d = dEdx.cols();
  dEdx += (*dEdf).middleCols(c - d, d);
#endif
}

void PairwiseRankLoss::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::vpairwise_rank_loss(fx.d.size(), margin, xs[0]->v, xs[1]->v, fx.v);
#else
  auto a = **xs[0];
  auto b = **xs[1];
  *fx = a.binaryExpr(b, FPairwiseRankLoss(margin));
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void PairwiseRankLoss::backward_impl(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i,
                                Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vpairwise_rank_loss_backward(dEdf.d.size(), (i == 0), fx.v, dEdf.v, dEdxi.v);
#else
  if (i == 0) {
    *dEdxi -= (*fx).binaryExpr(*dEdf, FRectifyBackward());
  } else {
    *dEdxi += (*fx).binaryExpr(*dEdf, FRectifyBackward());
  }
#endif
}

size_t Hinge::aux_storage_size() const {
  return dim.size() * sizeof(cnn::real);
}

void Hinge::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
  fx.m_device_id = xs[0]->m_device_id;
#ifdef HAVE_CUDA
  throw std::runtime_error("Hinge not yet implemented for CUDA");
#else
  auto x = **xs[0];
  const unsigned rows = x.rows();
  cnn::real y = 0;
  cnn::real* eloss = static_cast<cnn::real*>(aux_mem);
  const cnn::real mlystar = margin - x(*pelement);
  for (unsigned i = 0; i < rows; ++i) {
    if (*pelement != i) {
      eloss[i] = max<cnn::real>(0.f, mlystar + x(i));
      y += eloss[i];
    } else {
      eloss[i] = 0;
    }
  }
  fx.v[0] = y;
#endif
}

void Hinge::backward_impl(const vector<const Tensor*>& xs,
                       const Tensor& fx,
                       const Tensor& dEdf,
                       unsigned i,
                       Tensor& dEdxi) const {
  assert(i == 0);
#ifdef HAVE_CUDA
  throw std::runtime_error("Hinge not yet implemented for CUDA");
#else
  if (fx.v[0]) { // there was some loss
    const cnn::real d = dEdf.v[0];
    const unsigned rows = dEdxi.d.rows();
    const cnn::real* eloss = static_cast<const cnn::real*>(aux_mem);
    unsigned tne = 0;  // total number of errors
    for (unsigned i = 0; i < rows; ++i)
      if (eloss[i] > 0) {
        (*dEdxi)(i) += d;
        ++tne;
      }
    (*dEdxi)(*pelement) -= d * tne;
  }
#endif
}

void Identity::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  fx.d = xs[0]->d;
  fx.v = xs[0]->v;
  fx.m_device_id = xs[0]->m_device_id;
}

void Identity::backward_impl(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  *dEdxi += *dEdf;
}

void MaxPooling1D::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  cerr << "FIX IMPL5\n"; abort();
  fx.m_device_id = xs[0]->m_device_id;
#if 0
  assert(xs.size() == 1);
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  assert(x.cols() == 1);
  const unsigned fx_rows = x_rows / width;
  ind.resize(fx_rows);
  Tensor fx = Zero(Dim(fx_rows, 1));
  for (unsigned i = 0; i < fx_rows; ++i) {
    unsigned from = i * width;
    unsigned to = from + width;
    if (to > x_rows) to = x_rows;
    cnn::real best = x(from, 0);
    unsigned bestr = from;
    for (unsigned r = from + 1; r < to; ++r) {
      if (x(r, 0) > best) {
        best = x(r,0);
        bestr = r;
      }
    }
    ind[i] = bestr;
    fx(i, 0) = best;
  }
  return fx;
#endif
}

void MaxPooling1D::backward_impl(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
  cerr << "FIX IMPL6\n"; abort();
#if 0
  const Tensor& x = *xs.front();
  const unsigned x_rows = x.rows();
  Tensor dEdx = Zero(Dim(x_rows, 1));
  const unsigned fx_rows = x_rows / width;
  assert(fx_rows == ind.size());
  assert(fx_rows == dEdf.rows());
  for (unsigned i = 0; i < fx_rows; ++i)
    dEdx(ind[i], 0) = dEdf(i, 0);
  return dEdx;
#endif
}

void Softmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    assert(xs.size() == 1);
    fx.m_device_id = xs[0]->m_device_id;
#if HAVE_CUDA
    gpu::softmax(xs[0]->d.rows(), xs[0]->d.cols(), xs[0]->v, fx.v);
#else
    softmax<cnn::real>(xs[0]->d.rows(), xs[0]->d.cols(), xs[0]->v, fx.v, true);
#endif
}

void Softmax::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#if HAVE_CUDA 
    gpu::softmax_backward(fx.d.rows(), fx.d.cols(), fx.v, dEdf.v, dEdxi.v);
#else
    vector<cnn::real> off_diag_sum(fx.d.cols());
    int rows = fx.d.rows();

#pragma omp parallel for
    for (int k = 0; k < fx.d.cols(); k++)
    {
        off_diag_sum[k] = 0;
        for (unsigned r = 0; r < rows; r++)
            off_diag_sum[k] += fx.v[IDX2C(r, k, rows)] * dEdf.v[IDX2C(r, k, rows)];

        /// row-element-wise multiplication
        for (unsigned r = 0; r < rows; r++)
        {
            cnn::real d = dEdf.v[IDX2C(r, k, rows)]; 
            cnn::real t = fx.v[IDX2C(r, k, rows)];
            dEdxi.v[IDX2C(r, k, rows)] += (d - off_diag_sum[k]) * t;
        }
    }
#endif
}

/*
void PickNegLogSoftmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  if (xs[0]->d.cols() == 1) {
    logz = (cnn::real*)fxs->allocate(sizeof(cnn::real)*fx.d.batch_elems());
#if HAVE_CUDA
    if(pval) {
      gpu::pnlsoftmax(xs[0]->d.size(), *pval, xs[0]->v, fx.v, logz);
    } else {
      // TODO: It'd be nice to have a kernel that did all batches at once
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems());
      for(unsigned b = 0; b < pvals->size(); ++b)
        gpu::pnlsoftmax(xs[0]->d.batch_size(), (*pvals)[b], xs[0]->batch_ptr(b), fx.v+b, logz+b);
    }
#else
    if(pval) {
      auto x = **xs[0];
      *logz = logsumexp(x);
      fx.v[0] = *logz - x(*pval);
    } else {
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems());
      for(unsigned b = 0; b < pvals->size(); ++b) {
        auto x = xs[0]->batch_matrix(b);
        logz[b] = logsumexp(x);
        fx.v[b] = logz[b] - x((*pvals)[b]);
      }
    }
#endif
  } else {
    throw std::runtime_error("PickNegLogSoftmax::forward not yet implemented for multiple columns");
  }
}

there is probably a bug in using batch implementation
when runing on rnnlm2_cls, using pickenglogsoftmax actually get 0 PPL at epoch 4 and then crash, and that is impossible. 
when switching back to logsoftmax and do pick operation later, the results look reasonable with training PPL decreased to 40~50 at epoch 9. 
*/
/** comment the following out
void PickNegLogSoftmax::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  if (xs[0]->d.cols() == 1) {
#if HAVE_CUDA
    if(pval) {
      const auto elem = *pval;
      gpu::pnlsoftmax_backward(dEdxi.d.size(), elem, xs[0]->v, dEdf.v, logz, dEdxi.v);
    } else {
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems()); 
      // TODO: Again, it would be nice to do this with a single kernel
      for(unsigned b = 0; b < pvals->size(); ++b) {
        const auto elem = (*pvals)[b];
        gpu::pnlsoftmax_backward(dEdxi.d.batch_size(), elem, xs[0]->batch_ptr(b), dEdf.v+b, logz+b, dEdxi.batch_ptr(b));
      }
    }
#else
    if(pval) {
      const auto elem = *pval;
      const cnn::real err = dEdf.v[0];
      auto x = **xs[0];
      // logz is computed in the forward pass and cached
      *dEdxi += x.unaryExpr(FNegLogSoftmaxBackward(*logz, err));
      //*dEdxi += x.unaryExpr(scalar_nlsoftmax_backward_op<cnn::real>(*logz, err));
      (*dEdxi)(elem) -= err;
    } else {
      assert(pvals);
      assert(pvals->size() == fx.d.batch_elems()); 
      for(unsigned b = 0; b < pvals->size(); ++b) {
        const auto elem = (*pvals)[b];
        const cnn::real err = dEdf.v[b];
        auto x = xs[0]->batch_matrix(b);
        auto dEdxi_mat = dEdxi.batch_matrix(b);
        dEdxi_mat += x.unaryExpr(FNegLogSoftmaxBackward(logz[b], err));
        //dEdxi_mat += x.unaryExpr(scalar_nlsoftmax_backward_op<cnn::real>(logz[b], err));
        dEdxi_mat(elem) -= err;
      }
    }
#endif
  } else {
    throw std::runtime_error("PickNegLogSoftmax::backward not yet implemented for multiple columns");
  }
}
*/

size_t LogSoftmax::aux_storage_size() const {
    /// save space for softmax and a vector of nutt 
    int sz = dim.size() + dim.cols();
    return sz* sizeof(cnn::real);
}

void LogSoftmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::logsoftmax(xs[0]->d.rows(), xs[0]->d.cols(), xs[0]->v, fx.v);
#else
  logsoftmax<cnn::real>(xs[0]->d.rows(), xs[0]->d.cols(), xs[0]->v, fx.v, true);
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void LogSoftmax::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const 
{
    unsigned rows = dEdf.d.rows();
    unsigned cols = dEdf.d.cols();
    Tensor softmax(fx.d, static_cast<cnn::real*>(aux_mem)+cols, device_id);
    cnn::real* off_diag_sum = static_cast<cnn::real*>(aux_mem);

#if HAVE_CUDA
    gpu::logsoftmax_backward(xs[0]->d.rows(), xs[0]->d.cols(), fx.v, dEdf.v, dEdxi.v, softmax.v, off_diag_sum);
#else

    (*softmax).array() = (*fx).array().exp();

#pragma omp parallel for
    for (int k = 0; k < cols; k++)
    {
        off_diag_sum[k] = 0;
        for (unsigned r = 0; r < rows; r ++)
            off_diag_sum[k] += dEdf.v[IDX2C(r,k, rows)];

        /// row-element-wise multiplication
        for (unsigned r = 0; r < rows; r++)
        {
            softmax.v[IDX2C(r, k, rows)] *= off_diag_sum[k];
            dEdxi.v[IDX2C(r, k, rows)] += (dEdf.v[IDX2C(r, k, rows)] - softmax.v[IDX2C(r, k, rows)]);
        }
    }
#endif
}

template <class T>
EIGEN_STRONG_INLINE cnn::real logsumexp(const T& x, const vector<unsigned>& denom) {
  cnn::real m = x(denom[0],0);
  for (auto i : denom) {
    cnn::real r = x(i,0);
    if (r > m) m = r;
  }
  cnn::real z = 0;
  for (auto i : denom)
    z += expf(x(i,0) - m);
  return m + logf(z);
}

void RestrictedLogSoftmax::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("RestrictedLogSoftmax not yet implemented for CUDA");
#else
  // TODO create auxiliary mask with -infty's
  // and do usual LogSoftmax stuff
  assert(xs.size() == 1);
  assert(denom.size() > 0);
  auto x = **xs[0];
  assert(x.cols() == 1);
  const cnn::real logz = logsumexp(x, denom);
  TensorTools::Constant(fx, -numeric_limits<cnn::real>::infinity());
  for (auto i : denom)
    (*fx)(i,0) = x(i,0) - logz;
  if (denom.size() == 1) (*fx)(denom.front(), 0) = 0;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void RestrictedLogSoftmax::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  assert(i == 0);
#ifdef HAVE_CUDA
  throw std::runtime_error("RestrictedLogSoftmax not yet implemented for CUDA");
#else
  cnn::real z = 0;
  for (auto ind : denom)
    z += (*dEdf)(ind, 0);
  for (auto ind : denom)
    (*dEdxi)(ind, 0) += (*dEdf)(ind, 0) - expf((*fx)(ind, 0)) * z;
#endif
}

// x_1 is a vector
// y = (x_1)_{*pval}
void PickElement::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  CUDA_CHECK(cudaMemcpyAsync(&fx.v[0], &xs[0]->v[*pval], sizeof(cnn::real), cudaMemcpyDeviceToDevice));
#else
  auto x = **xs[0];
  fx.v[0] = x(*pval);
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

// derivative is 0 in all dimensions except 1 for the selected element
void PickElement::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                    unsigned i,
                    Tensor& dEdxi) const {
  assert(i == 0);
#if HAVE_CUDA
  if (sizeof(cnn::real) == sizeof(float))
      CUBLAS_CHECK(cublasSaxpy(cublas_handle, 1, reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(dEdf.v), 1, reinterpret_cast<float*>(&dEdxi.v[*pval]), 1));
  else if (sizeof(cnn::real) == sizeof(double))
      CUBLAS_CHECK(cublasDaxpy(cublas_handle, 1, reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(dEdf.v), 1, reinterpret_cast<double*>(&dEdxi.v[*pval]), 1));
#else
  (*dEdxi)(*pval) += dEdf.v[0];
#endif
}

// x_1 is a vector
// y = (x_1)[start:end]
// slice of vector from index start (inclusive) to index end (exclusive)
void PickRange::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    assert(xs.size() == 1);
    auto x = **xs[0];
    assert(x.cols() == 1);
    assert(start >= 0);
    assert(end <= x.rows());
    assert(start < end);
    assert(int(fx.d.rows()) == int(end - start));
#if HAVE_CUDA
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[0], &xs[0]->v[start], sizeof(cnn::real) * (end-start), cudaMemcpyDeviceToDevice));
#else
    (*fx) = x.block(start, 0, end - start, 1);
#endif
    fx.m_device_id = xs[0]->m_device_id;
}

// derivative is 0 in all dimensions except the slice range
void PickRange::backward_impl(const vector<const Tensor*>& xs,
    const Tensor& fx,
    const Tensor& dEdf,
    unsigned i,
    Tensor& dEdxi) const {
    assert(i == 0);
    assert(int(dEdf.d.rows()) == int(end - start));
    assert(dEdf.d.cols() == 1);
#if HAVE_CUDA
    if (sizeof(cnn::real) == sizeof(float))
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, end - start, reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(dEdf.v), 1, reinterpret_cast<float*>(&dEdxi.v[start]), 1));
    else if (sizeof(cnn::real) == sizeof(double))
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, end - start, reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(dEdf.v), 1, reinterpret_cast<double*>(&dEdxi.v[start]), 1));
#else
    (*dEdxi).block(start, 0, end - start, 1) += (*dEdf);
#endif
}

// x_1 is a matrix
// y = x_1[start_column: start_column + rows * (end_column - start_column)]
// (start_column inclusive, end_column exclusive)
void ColumnSlices::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    assert(xs.size() == 1);
    auto x = **xs[0];
    assert(x.cols() >= 1);
    assert(start_column >= 0);
    assert(rows <= x.rows());
    assert(start_column < end_column);
    assert(int(fx.d.rows()) == int(rows));
#if HAVE_CUDA
    int start = start_column*rows;
    CUDA_CHECK(cudaMemcpyAsync(&fx.v[0], &xs[0]->v[start], sizeof(cnn::real) * rows * (end_column - start_column), cudaMemcpyDeviceToDevice));
#else
    (*fx) = x.block(0, start_column, rows, end_column - start_column);
#endif
    fx.m_device_id = xs[0]->m_device_id;
}

// derivative is 0 in all dimensions except the slice range
void ColumnSlices::backward_impl(const vector<const Tensor*>& xs,
    const Tensor& fx,
    const Tensor& dEdf,
    unsigned i,
    Tensor& dEdxi) const {
    assert(i == 0);
    assert(int(dEdf.d.rows()) == int(rows));
    assert(dEdf.d.cols() == end_column - start_column);
#if HAVE_CUDA
    if (sizeof(cnn::real) == sizeof(float))
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, rows * (end_column - start_column), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(dEdf.v), 1, reinterpret_cast<float*>(&dEdxi.v[rows * start_column]), 1));
    else if (sizeof(cnn::real) == sizeof(double))
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, rows * (end_column - start_column), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(dEdf.v), 1, reinterpret_cast<double*>(&dEdxi.v[rows * start_column]), 1));
#else
    (*dEdxi).block(0, start_column, rows, end_column -start_column) += (*dEdf);
#endif
}

#if HAVE_CUDA
inline void CUDAMatrixMultiply(const Tensor& l, const Tensor& r, Tensor& y, const cnn::real* acc_scalar) {
  // if (r.d.ndims() == 1 || r.d.cols() == 1) {
  //   CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_N, l.d.rows(), l.d.cols(),
  //              kSCALAR_ONE, l.v, l.d.rows(), r.v, 1, acc_scalar, y.v, 1));
  // } else {
  //   CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
  //         y.d.rows(), y.d.cols(), l.d.cols(),
  //         kSCALAR_ONE,
  //         l.v, l.d.rows(),
  //         r.v, r.d.rows(),
  //         acc_scalar, y.v, y.d.rows()));
  // }
    // If the left side has one batch, multiply by columns
    // [x, z, b] = [x, y] * [y, z, b]
    // -> [x, z*b] = [x, y], [y, z*b]
#ifdef USE_DOUBLE
   CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          y.d.rows(), y.d.cols(), l.d.cols(),
          kSCALAR_ONE,
          l.v, l.d.rows(),
          r.v, r.d.rows(),
          acc_scalar,
          y.v, y.d.rows()));
#else
   CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
          y.d.rows(), y.d.cols(), l.d.cols(),
          kSCALAR_ONE,
          l.v, l.d.rows(),
          r.v, r.d.rows(),
          acc_scalar, y.v, y.d.rows()));
#endif
}
#endif

void MatrixMultiply::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  // fx = 0*fx + xs[0] * xs[1]
  CUDAMatrixMultiply(*xs[0], *xs[1], fx, kSCALAR_ZERO);
#else
  assert(fx.d.bd == 1); 
  // If the left side has one batch, multiply by columns
  // [x, z, b] = [x, y] * [y, z, b]
  // -> [x, z*b] = [x, y], [y, z*b]
  fx.colbatch_matrix().noalias() = **xs[0] * xs[1]->colbatch_matrix();
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void MatrixMultiply::backward_impl(const vector<const Tensor*>& xs,
                                const Tensor& fx,
                                const Tensor& dEdf,
                                unsigned i,
                                Tensor& dEdxi) const {
  assert(i < 2);
#if HAVE_CUDA
  if (i == 0) {
#ifdef USE_DOUBLE
      CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
        kSCALAR_ONE,
        dEdf.v, dEdf.d.rows(),
        xs[1]->v, xs[1]->d.rows(),
        kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#else
      CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
          dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
          kSCALAR_ONE,
          dEdf.v, dEdf.d.rows(),
          xs[1]->v, xs[1]->d.rows(),
          kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#endif
  }
  else {
#ifdef USE_DOUBLE
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        dEdxi.d.rows(), dEdxi.d.cols(), xs[0]->d.rows(),
        kSCALAR_ONE,
        xs[0]->v, xs[0]->d.rows(),
        dEdf.v, dEdf.d.rows(),
        kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#else
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        dEdxi.d.rows(), dEdxi.d.cols(), xs[0]->d.rows(),
        kSCALAR_ONE,
        xs[0]->v, xs[0]->d.rows(),
        dEdf.v, dEdf.d.rows(),
        kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#endif
  }
#else
  if (i == 0) {
    dEdxi.colbatch_matrix().noalias() += dEdf.colbatch_matrix() * xs[1]->colbatch_matrix().transpose();
  } else {
    dEdxi.colbatch_matrix().noalias() += xs[0]->colbatch_matrix().transpose() * dEdf.colbatch_matrix();
  }
#endif
}

void CwiseQuotient::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("CwiseQuotient::forward not yet implemented for CUDA");
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  *fx = x1.cwiseQuotient(x2);
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void CwiseQuotient::backward_impl(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("CwiseQuotient::backward not yet implemented for CUDA");
#else
  if (i == 0) {
    auto x2 = **xs[1];
    *dEdxi += (*dEdf).cwiseQuotient(x2);
  } else { // i = 1
    auto x1 = **xs[0];
    auto x2 = **xs[1];
    *dEdxi -= (*dEdf).cwiseQuotient(x2.cwiseProduct(x2)).cwiseProduct(x1);
  }
#endif
}

void CwiseMultiply::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  gpu::vcwise_product(fx.d.size(), xs[0]->v, xs[1]->v, fx.v);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  *fx = x1.cwiseProduct(x2);
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void CwiseMultiply::backward_impl(const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  assert(i < 2);
  if (i == 0) {
#if HAVE_CUDA
    gpu::vcwise_product_backward(fx.d.size(), dEdf.v, xs[1]->v, dEdxi.v);
#else
    auto x2 = **xs[1];
    *dEdxi += (*dEdf).cwiseProduct(x2);
#endif
  } else {
#if HAVE_CUDA
    gpu::vcwise_product_backward(fx.d.size(), dEdf.v, xs[0]->v, dEdxi.v);
#else
    auto x1 = **xs[0];
    *dEdxi += (*dEdf).cwiseProduct(x1);
#endif
  }
}

void AffineTransform::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() % 2 == 1);
#if HAVE_CUDA
  for (unsigned i = 1; i < xs.size(); i += 2)
    // fx = (acc_sclar)*fx + xs[0] * xs[1]
    CUDAMatrixMultiply(*xs[i], *xs[i + 1], fx, (i == 1) ? kSCALAR_ZERO : kSCALAR_ONE);
  assert(fx.d.bd == 1);
  assert(xs[0]->d.bd == 1);
  if (sizeof(cnn::real) == sizeof(float))
      CUBLAS_CHECK(cublasSaxpy(cublas_handle, fx.d.size(), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(xs[0]->v), 1, reinterpret_cast<float*>(fx.v), 1));
  else if (sizeof(cnn::real) == sizeof(double))
      CUBLAS_CHECK(cublasDaxpy(cublas_handle, fx.d.size(), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(xs[0]->v), 1, reinterpret_cast<double*>(fx.v), 1));
#else
  assert(fx.d.bd == 1);
  // Add, using broadcasting or not
  fx.colbatch_matrix().noalias() = xs[0]->colbatch_matrix();
  
  // Multiply
  for (unsigned i = 1; i < xs.size(); i += 2) {
    fx.colbatch_matrix().noalias() += **xs[i] * xs[i+1]->colbatch_matrix();
  }
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void AffineTransform::backward_impl(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  assert(i < xs.size());
  if (i == 0) { // bias term
#if HAVE_CUDA
    if (sizeof(cnn::real) == sizeof(float))
        CUBLAS_CHECK(cublasSaxpy(cublas_handle, dEdxi.d.size(), reinterpret_cast<float*>(kSCALAR_ONE), reinterpret_cast<float*>(dEdf.v), 1, reinterpret_cast<float*>(dEdxi.v), 1));
    else if (sizeof(cnn::real) == sizeof(double))
        CUBLAS_CHECK(cublasDaxpy(cublas_handle, dEdxi.d.size(), reinterpret_cast<double*>(kSCALAR_ONE), reinterpret_cast<double*>(dEdf.v), 1, reinterpret_cast<double*>(dEdxi.v), 1));
#else
    assert(fx.d.bd == 1);
    // Add, using broadcasting or not
    dEdxi.colbatch_matrix().noalias() += dEdf.colbatch_matrix();
#endif
  } else if (i % 2 == 1) { // left argument of matrix multiply
#if HAVE_CUDA
#ifdef USE_DOUBLE
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
        kSCALAR_ONE,
        dEdf.v, dEdf.d.rows(),
        xs[i + 1]->v, xs[i + 1]->d.rows(),
        kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#else
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
        kSCALAR_ONE,
        dEdf.v, dEdf.d.rows(),
        xs[i + 1]->v, xs[i + 1]->d.rows(),
        kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#endif
#else
    dEdxi.colbatch_matrix().noalias() += dEdf.colbatch_matrix() * xs[i+1]->colbatch_matrix().transpose();
#endif
  } else {  // right argument of matrix multiply
#if HAVE_CUDA
#ifdef USE_DOUBLE
    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        dEdxi.d.rows(), dEdxi.d.cols(), xs[i - 1]->d.rows(),
        kSCALAR_ONE,
        xs[i - 1]->v, xs[i - 1]->d.rows(),
        dEdf.v, xs[i - 1]->d.rows(),
        kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#else
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        dEdxi.d.rows(), dEdxi.d.cols(), xs[i - 1]->d.rows(),
        kSCALAR_ONE,
        xs[i - 1]->v, xs[i - 1]->d.rows(),
        dEdf.v, xs[i - 1]->d.rows(),
        kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
#endif
#else
    dEdxi.colbatch_matrix().noalias() += (**xs[i-1]).transpose() * dEdf.colbatch_matrix();
#endif
  }
}

void Negate::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vnegate(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = -x;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void Negate::backward_impl(const vector<const Tensor*>& xs,
                      const Tensor& fx,
                      const Tensor& dEdf,
                      unsigned i,
                      Tensor& dEdxi) const {
  assert(i == 0);
#if HAVE_CUDA
  gpu::vnegate_backward(fx.d.size(), dEdf.v, dEdxi.v);
#else
  *dEdxi -= *dEdf;
#endif
}

void Rectify::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vrelu(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(FRectify());
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void Rectify::backward_impl(const vector<const Tensor*>& xs,
                         const Tensor& fx,
                         const Tensor& dEdf,
                         unsigned i,
                         Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vrelu_backward(fx.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, FRectifyBackward());
#endif
}

void ExponentialLinearUnits::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    assert(xs.size() == 1);
#if HAVE_CUDA
    gpu::vexponential_linear_units(fx.d.size(), xs[0]->v, scale, fx.v);
#else
    auto x = **xs[0];
    *fx = x.unaryExpr(FExponentialLinearUnits(scale));
#endif
    fx.m_device_id = xs[0]->m_device_id;
}

void ExponentialLinearUnits::backward_impl(const vector<const Tensor*>& xs,
    const Tensor& fx,
    const Tensor& dEdf,
    unsigned i,
    Tensor& dEdxi) const {
#if HAVE_CUDA
    gpu::vexponential_linear_units_backward(fx.d.size(), fx.v, dEdf.v, scale, dEdxi.v);
#else
    *dEdxi += (*fx).binaryExpr(*dEdf, FExponentialLinearUnitsBackward(scale));
#endif
}

void HuberDistance::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("HuberDistance not yet implemented for CUDA");
#else
  auto x = *xs[0];
  auto y = *xs[1];
  const FHuberForward fhf(d);
  const size_t s = x.d.size();
  cnn::real dist = 0;
  for (size_t i = 0; i < s; ++i)
    dist += fhf(x.v[i] - y.v[i]);
  fx.v[0] = dist;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void HuberDistance::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("HuberDistance not yet implemented for CUDA");
#else
  auto x = **xs[i];
  auto y = **xs[1-i];
  *dEdxi += (x - y).unaryExpr(FHuberBackward(d, dEdf.v[0]));
#endif
}

void L1Distance::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("L1Distance not yet implemented for CUDA");
#else
  auto x = **xs[0];
  auto y = **xs[1];
  fx.v[0] = (x - y).lpNorm<1>();
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void L1Distance::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(i < 2);
#ifdef HAVE_CUDA
  throw std::runtime_error("L1Distance not yet implemented for CUDA");
#else
  auto x = **xs[i];
  auto y = **xs[1-i];
  *dEdxi += (x - y).unaryExpr(FL1Backward(dEdf.v[0]));
#endif
}

void PoissonRegressionLoss::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("PoissonRegressionLoss not yet implemented for CUDA");
#else
  const auto y = *pty;
  const auto z = lgamma(y + 1);
  const auto x = xs[0]->v[0];
  fx.v[0] = expf(x) + z - y * x;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void PoissonRegressionLoss::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("PoissonRegressionLoss not yet implemented for CUDA");
#else
  const auto x = xs[0]->v[0];
  const auto y = *pty;
  auto& dEdx = dEdxi.v[0];
  dEdx += expf(x) - y;
#endif
}

void SquaredEuclideanDistance::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 2);
#if HAVE_CUDA
  gpu::sqeucdist(xs[0]->d.size(), xs[0]->v, xs[1]->v, fx.v);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  fx.v[0] = (x1 - x2).squaredNorm();
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void SquaredEuclideanDistance::backward_impl(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
  assert(i < 2);
#if HAVE_CUDA
  gpu::sqeucdist_backward(xs[0]->d.size(), dEdf.v, xs[0]->v, xs[1]->v, dEdxi.v, i);
#else
  auto x1 = **xs[0];
  auto x2 = **xs[1];
  cnn::real scale = dEdf.v[0] * 2;
  if (i == 1) scale = -scale;
  *dEdxi += scale * (x1 - x2);
#endif
}

void LogisticSigmoid::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#if HAVE_CUDA
  gpu::vlogistic(fx.d.size(), xs[0]->v, fx.v);
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(scalar_logistic_sigmoid_op<cnn::real>());
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void LogisticSigmoid::backward_impl(const vector<const Tensor*>& xs,
                                 const Tensor& fx,
                                 const Tensor& dEdf,
                                 unsigned i,
                                 Tensor& dEdxi) const {
#if HAVE_CUDA
  gpu::vlogistic_backward(dEdf.d.size(), fx.v, dEdf.v, dEdxi.v);
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, scalar_logistic_sigmoid_backward_op<cnn::real>());
#endif
}

void SoftSign::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 1);
#ifdef HAVE_CUDA
  throw std::runtime_error("SoftSign not yet implemented for CUDA");
#else
  auto x = **xs[0];
  *fx = x.unaryExpr(FSoftSign());
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void SoftSign::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("SoftSign not yet implemented for CUDA");
#else
  *dEdxi += (*fx).binaryExpr(*dEdf, FSoftSignBackward());
#endif
}

void BinaryLogLoss::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BinaryLogLoss not yet implemented for CUDA");
#else
  auto x = *xs[0];
  auto y = *xs[1];
  FBinaryLogLoss bll;
  const size_t s = x.d.size();
  cnn::real dist = 0;
  for (size_t i = 0; i < s; ++i)
    dist += bll(x.v[i], y.v[i]);
  fx.v[0] = dist;
#endif
  fx.m_device_id = xs[0]->m_device_id;
}

void BinaryLogLoss::backward_impl(const vector<const Tensor*>& xs,
                  const Tensor& fx,
                  const Tensor& dEdf,
                  unsigned i,
                  Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("BinaryLogLoss not yet implemented for CUDA");
#else
  *dEdxi += (**xs[i]).binaryExpr(**xs[1-i], FBinaryLogLossBackward(dEdf.v[0]));
#endif
}

string Zeroes::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "zeroes(" << dim << ')';
  return s.str();
}

Dim Zeroes::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void Zeroes::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  if (fx.m_device_id < 0)
      memset(fx.v, 0, dim.size() * sizeof(cnn::real));
  else{
#if HAVE_CUDA
      cudaMemsetAsync(fx.v, 0, dim.size() * sizeof(cnn::real));
#else
      memset(fx.v, 0, dim.size() * sizeof(cnn::real));
#endif
  }
  fx.m_device_id = xs[0]->m_device_id;
}

void Zeroes::backward_impl(const vector<const Tensor*>& xs,
                           const Tensor& fx,
                           const Tensor& dEdf,
                           unsigned i,
                           Tensor& dEdxi) const {
  throw std::runtime_error("Called backward() on an arity 0 node");
}

} // namespace cnn
