#ifndef CNN_GPU_FUNCTORS_H
#define CNN_GPU_FUNCTORS_H

#include <cstdint>
#include <cnn/macros.h>
using namespace std;
#if HAVE_CUDA
#define CNN_DEVICE_FUNC __device__
#define CNN_DEVICE_MIN -1.175494351e-38f
#else
#include <boost/math/special_functions/digamma.hpp>
#define CNN_DEVICE_FUNC
#define CNN_DEVICE_MIN -1.175494351e-38f
#endif
#include <cnn/macros.h>

// these functions are used both in CPU and in GPU computation
// this file may be compiled with NVCC or a standard C++ tool.
// if you need a new elementwise (nullary, unary, binary...)
// functor, this is the place for it
//
// note: also see xfunctors.h - functors implemented there can
// use Eigen's internal support for vectorized operations which
// can give faster performance on some hardware

#define cast_uint32_t static_cast<uint32_t>

// THIS CODE IS BROKEN- sometimes it returns NaN
// it is commented out for this reason
template<class ElemType>
static inline ElemType fastpow2 (ElemType p) {
  ElemType offset = (p < 0) ? 1.0f : 0.0f;
  ElemType clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  cnn::real z = clipp - w + offset;
  union { uint32_t i; cnn::real f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

static CNN_DEVICE_FUNC inline cnn::real fastexp(cnn::real p) {
    return (sizeof(cnn::real) == sizeof(float)) ? expf(p) : exp(p);
}

#define CNN_EXPF fastexp

#if defined(__GNU_LIBRARY__) && (__GLIBC__ == 2) && (__GLIBC_MINOR__ < 14) && !defined(HAVE_CUDA)
#define USE_FASTEXP
#else
#undef USE_FASTEXP
#endif

namespace cnn {

struct FHuberForward {
FHuberForward(cnn::real c) : c(c) {}
CNN_DEVICE_FUNC inline cnn::real operator()(cnn::real x) const {
    const cnn::real a = fabs(x);
    return (a < c) ? x*x : c*(2*a - c);
    }
    const cnn::real c;
};

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

struct FL1Backward {
  FL1Backward(cnn::real d) : d(d) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real & x) const {
    return sgn(x) * d;
  }
  const cnn::real d;
};

struct FHuberBackward {
  FHuberBackward(cnn::real c, cnn::real dEdf) : c(c), d(dEdf) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real & x) const {
    const cnn::real a = fabs(x);
    return (2 * d) * ((a < c) ? x : c * sgn(x));
  }
  const cnn::real c;
  const cnn::real d;
};

struct FSubtract {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &a, const cnn::real &b) const {
        return a - b;
    }
};

struct FProduct {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &a, const cnn::real &b) const {
        return a * b;
    }
};

struct FQuotient {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &a, const cnn::real &b) const {
    return a / b;
  }
};

struct FSquare {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
        return x * x;
    }
};

struct FConstantMultiply{
    FConstantMultiply(cnn::real c) : c(c) {}
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
        return c * x;
    }
    cnn::real c;
};

struct FConstantPlus {
  FConstantPlus(cnn::real c) : c(c) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
    return c + x;
  }
  cnn::real c;
};

struct FConstantMinus {
  FConstantMinus(cnn::real c) : c(c) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
    return c - x;
  }
  cnn::real c;
};

struct FCopy {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
        return x;
    }
};

struct FNegate {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
    return -x;
  }
};

struct FErf {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
    return erff(x);
  }
};

struct FTanh {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
#ifdef FAST_TANH
    cnn::real x2 = x * x;
    cnn::real a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    cnn::real b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    return a / b;
#else
    return (sizeof(cnn::real) == sizeof(float))?tanhf(x):tanh(x);
#endif
  }
};

struct FMaxBackwardInv {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &u, const cnn::real &d) const {
    return (1.f - u) * d;
  }
};

struct FSqrtBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(cnn::real t, cnn::real d) const {
    return d / (2.f * t);
  }
};

struct FErfBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(cnn::real x, cnn::real d) const {
    return 1.1283791670955125738961589f * expf(-x * x) * d;
  }
};

struct FTanhBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
    return (1.f - t * t) * d;
  }
};

struct FLogBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real & t, const cnn::real& d) const {
    return (1.f / t) * d;
  }
};

struct FPairwiseRankLoss {
  FPairwiseRankLoss(cnn::real m) : margin(m) {}
  CNN_DEVICE_FUNC cnn::real operator()(const cnn::real &a, const cnn::real &b) const {
    cnn::real d = margin - a + b;
    return d > 0.f ? d : 0.f;
  }
  cnn::real margin;
};

struct FRectifyBackward {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
        return (t) ? d : 0.f;
    }
};

struct FExponentialLinearUnitsBackward {
    FExponentialLinearUnitsBackward(cnn::real m) : a(m) {}
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
        return (t) ? d : d * (t + a);
    }
    cnn::real a; /// scale in the negative input part
};

struct FRectifyNegateBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
    return (t) ? -d : 0.f;
  }
};

struct FSoftmaxNormalize {
  explicit FSoftmaxNormalize(cnn::real logz) : logz(logz) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
    return fastexp(x - logz);
  }
  cnn::real logz;
};

struct FSoftmaxBackward {
  explicit FSoftmaxBackward(cnn::real off_diag_sum) : off_diag_sum(off_diag_sum) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
    return (off_diag_sum + d) * t;
  }
  cnn::real off_diag_sum;
};
struct FLogGammaBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(cnn::real x, cnn::real d) const {
#ifndef HAVE_CUDA
    return boost::math::digamma(x) * d;
#else
    assert(false); // Not supported on GPUs?
    return 0;
#endif
  }
};

struct FNegLogSoftmaxBackward {
  FNegLogSoftmaxBackward(cnn::real lz, cnn::real err) : logz(lz), d(err) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t) const {
    return CNN_EXPF(t - logz) * d;
  }
  cnn::real logz;
  cnn::real d;
};

struct FPtrNegLogSoftmaxBackward {
  FPtrNegLogSoftmaxBackward(const cnn::real* lz, const cnn::real* err) : logz(lz), d(err) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t) const {
    return CNN_EXPF(t - *logz) * *d;
  }
  const cnn::real* logz;
  const cnn::real* d;
};

struct FLogSoftmaxNormalize {
  explicit FLogSoftmaxNormalize(cnn::real logz) : logz(logz) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
    return x - logz;
  }
  cnn::real logz;
};

struct FWeightedError {
  CNN_DEVICE_FUNC cnn::real operator()(const cnn::real & t, const cnn::real &d) const {
    return CNN_EXPF(t) * d / CNN_EXPF(t);
  }
};

struct FLogSoftmaxBackward {
  explicit FLogSoftmaxBackward(cnn::real off_diag_sum) : off_diag_sum(off_diag_sum) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
      return off_diag_sum * fastexp(t) + d;
   }
  cnn::real off_diag_sum;
};

struct FRectify {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
        return (x > 0.f) ? x : 0.f;
    }
};

struct FExponentialLinearUnits {
    explicit FExponentialLinearUnits(cnn::real scale) : a(scale) {}
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
        return (x > 0.f) ? x : a * (exp(x) - 1.0f);
    }
    cnn::real a; /// scale in the negative input part
};

struct FSoftSign {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
    return x / (1.f + (x < 0.f ? -x : x));
  }
};

struct FSoftSignBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
    cnn::real a = 1.f - (t < 0.f ? -t : t);
    return a * a * d;
  }
};

struct FLogisticSigmoid {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
      return 1.f / (1.f + CNN_EXPF(-x));
  }
};

struct FLogisticSigmoidBackward {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &t, const cnn::real &d) const {
    return (1.f - t) * t * d;
  }
};

struct FSqDist {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &a, const cnn::real &b) const {
    cnn::real d = a - b;
    return d * d;
  }
};

struct FExp {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
        return CNN_EXPF(x); 
    }
};

struct FLog {
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x) const {
        if (x < 1e-25) return LZERO;
        return (sizeof(cnn::real) == sizeof(float)) ? logf(x) : log(x);
    }
};

struct FEuclideanBackward {
  FEuclideanBackward(int i, const cnn::real* s) : i(i), scalar(s) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &a, const cnn::real &b) const {
    return (i == 0 ? 2.f : -2.f) * (*scalar) * (a - b);
  }
  int i;
  const cnn::real* scalar;
};

struct FL2SGDUpdate {
    FL2SGDUpdate(cnn::real l, cnn::real s) : lambda(l), scale(-s) {}
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x, const cnn::real &g) const {
        return scale * g - x * lambda;
    }
    cnn::real lambda;
    cnn::real scale;
};

struct FL2SGDMomentumUpdate {
    FL2SGDMomentumUpdate(cnn::real l, cnn::real s, cnn::real m) : lambda(l), scale(-s), momentum(m) {}
    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x, const cnn::real &g, cnn::real &v) {
        v = momentum * v + scale * g;
        return v - x * lambda;
    }
    cnn::real lambda;
    cnn::real scale;
    cnn::real momentum;
};

struct FBinaryLogLoss {
  CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real &x, const cnn::real &x_true) const {
    cnn::real x_tmp = x;
    if (x_true == 1.f) {
      if (x == 0.f) x_tmp = CNN_DEVICE_MIN;
      return -1.f * x_true * log(x_tmp);
    }
    else if (x_true == 0.f) {
      if (x == 1.f) x_tmp = CNN_DEVICE_MIN;
      return (x_true - 1.f) * log1p(-x_tmp);
    }
    else {
      if (x == 0.f) x_tmp = CNN_DEVICE_MIN;
      if (x == 1.f) x_tmp = CNN_DEVICE_MIN;
      return -1.f * (x_true * log(x_tmp) + (1.f - x_true) * log1p(-x_tmp));
    }
  }
};

struct FBinaryLogLossBackward {
  explicit FBinaryLogLossBackward(cnn::real d) : d(d) {}
  CNN_DEVICE_FUNC inline cnn::real operator()(cnn::real x, cnn::real x_true) const {
    cnn::real x_tmp = x;
    if (x == x_true) return 0;
    if (x == 0.f) x_tmp = CNN_DEVICE_MIN;
    if (x == 1.f) x_tmp = 0.9999999f;
    if (x_true == 1.f) {
      return d * -x_true / x_tmp;
    } else if (x_true == 0.f) {
      return d * (1.f - x_true) / (1.f - x_tmp);
	}
    return d * ((1.f - x_true) / (1.f - x_tmp) + (-x_true / x_tmp));
  }
  cnn::real d;
};

struct scale_functor
{
    const cnn::real a;

    scale_functor(cnn::real _a) : a(_a) {}

    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real& x) const
    {
        return a * x;
    }
};

struct saxpy_functor
{
    const cnn::real a;

    saxpy_functor(cnn::real _a) : a(_a) {}

    CNN_DEVICE_FUNC inline cnn::real operator()(const cnn::real& x, const cnn::real& y) const
    {
        return a * x + y;
    }
};

template <class ElemType>
void logsoftmax(int row, int col, const ElemType* a, ElemType* v, const bool isColWise)
{

    if (isColWise)
    {
#pragma omp parallel for
        for (int j = 0; j < col; j++)
        {
            // we need to extract max before applying exp to avoid overflow
            ElemType maxV = a[IDX2C(0, j, row)];
            for (int i = 1; i < row; i++)
                maxV = (maxV > a[IDX2C(i, j, row)]) ? maxV : a[IDX2C(i, j, row)];

            ElemType sum = 0;
            for (int i = 0; i < row; i++)
                sum += exp(v[IDX2C(i, j, row)] = a[IDX2C(i, j, row)] - maxV);
            sum = log(sum);
            for (int i = 0; i < row; i++)
                v[IDX2C(i, j, row)] -= sum;
        }
    }
    else
    {
        throw("not supported for row-major");
    }
}

template <class ElemType>
void softmax(int row, int col, const ElemType* a, ElemType* v, const bool isColWise)
{

    if (isColWise)
    {
#pragma omp parallel for
        for (int j = 0; j < col; j++)
        {
            // we need to extract max before applying exp to avoid overflow
            ElemType maxV = a[IDX2C(0, j, row)];
            for (int i = 1; i < row; i++)
                maxV = (maxV > a[IDX2C(i, j, row)]) ? maxV : a[IDX2C(i, j, row)];

            ElemType sum = 0;
            for (int i = 0; i < row; i++)
                sum += exp(v[IDX2C(i, j, row)] = a[IDX2C(i, j, row)] - maxV);
            sum = log(sum);
            for (int i = 0; i < row; i++)
            {
                ElemType tmp = v[IDX2C(i, j, row)] - sum;
                v[IDX2C(i, j, row)] = exp(tmp);
            }
        }
    }
    else
    {
        throw("not supported for row-major");
    }
}

} // namespace cnn

#endif
