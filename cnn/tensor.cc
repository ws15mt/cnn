#include "cnn/tensor.h"

#include <random>
#include <vector>
#include <cstring>

#if HAVE_CUDA
#include "cnn/cuda.h"
#endif

using namespace std;

namespace cnn {

    ostream& operator<<(ostream& os, const Tensor& t) {
    #if HAVE_CUDA
      vector<cnn::real> vt = as_vector(t);
      Eigen::Map<EMatrix> m(&vt[0], t.d.rows(), t.d.cols());
      os << m;
    #else
      os << (*t);
    #endif
      return os;
    }

    cnn::real as_scalar(const Tensor& t) {
      assert(t.d.size() == 1);
    #if HAVE_CUDA
      cnn::real res;
      CUDA_CHECK(cudaMemcpy(&res, t.v, sizeof(cnn::real), cudaMemcpyDeviceToHost));
      return res;
    #else
      return t.v[0];
    #endif
    }

    vector<cnn::real> as_vector(const Tensor& v) {
      vector<cnn::real> res(v.d.size());
    #if HAVE_CUDA
      CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(cnn::real) * res.size(), cudaMemcpyDeviceToHost));
    #else
      memcpy(&res[0], v.v, sizeof(cnn::real) * res.size());
    #endif
      return res;
    }

    vector<cnn::real> as_vector(int nsize, const cnn::real* v) {
        vector<cnn::real> res(nsize);
#if HAVE_CUDA
        CUDA_CHECK(cudaMemcpy(&res[0], v, sizeof(cnn::real) * res.size(), cudaMemcpyDeviceToHost));
#else
        memcpy(&res[0], v, sizeof(cnn::real) * res.size());
#endif
        return res;
    }

    cnn::real TensorTools::AccessElement(const Tensor& v, int index) {
    #if HAVE_CUDA
      cnn::real ret;
      cudaMemcpy(&ret, &v.v[index], sizeof(cnn::real), cudaMemcpyDeviceToHost);
      return ret;
    #else
      return v.v[index];
    #endif
    }

    cnn::real TensorTools::AccessElement(const Tensor& v, const Dim& index) {
    #if HAVE_CUDA
      abort();
    #else
      return (*v)(index[0], index[1]);
    #endif
    }

    void TensorTools::SetElement(const Tensor& v, int index, cnn::real value) {
    #if HAVE_CUDA
      cudaMemcpyAsync(&v.v[index], &value, sizeof(cnn::real), cudaMemcpyHostToDevice);
    #else
      v.v[index] = value;
    #endif
    }

    void TensorTools::SetElements(const Tensor& v, const vector<cnn::real>& vec) {
    #if HAVE_CUDA
      cudaMemcpyAsync(v.v, &vec[0], sizeof(cnn::real) * vec.size(), cudaMemcpyHostToDevice);
    #else
      memcpy(v.v, &vec[0], sizeof(cnn::real) * vec.size());
    #endif
    }

    void TensorTools::CopyElements(const Tensor& v, const Tensor& v_src) {
    #if HAVE_CUDA
      cudaMemcpyAsync(v.v, v_src.v, sizeof(cnn::real) * v.d.size(), cudaMemcpyDeviceToDevice);
    #else
      memcpy(v.v, v_src.v, sizeof(cnn::real) * v.d.size());
    #endif
    }

    void TensorTools::PushElementsToMemory(int& size, const int buf_size, cnn::real* v, const Tensor& v_src) {
        if (size + v_src.d.size() > buf_size)
            runtime_error("PushElementsToMemory : memory exhausted");
#if HAVE_CUDA
        /// save to the current available location
        cudaMemcpyAsync(v+size, v_src.v, sizeof(cnn::real) * v_src.d.size(), cudaMemcpyDeviceToDevice);
#else
        memcpy(v+size, v_src.v, sizeof(cnn::real) * v_src.d.size());
#endif
        size += v_src.d.size();
    }

    void TensorTools::Constant(Tensor& d, cnn::real c) {
    #if HAVE_CUDA
      if (!c) {
        CUDA_CHECK(cudaMemsetAsync(d.v, 0, d.d.size() * sizeof(cnn::real)));
      } else {
        fill(d.v, d.v + d.d.size(), c);
      }
    #else
      if (!c) {
        memset(d.v, c, d.d.size() * sizeof(cnn::real));
      } else {
        fill(d.v, d.v + d.d.size(), c);
      }
    #endif
    }

    void TensorTools::Zero(Tensor& d) {
      Constant(d, 0);
    }

    void TensorTools::Randomize(Tensor& val, cnn::real scale) {
      uniform_real_distribution<cnn::real> distribution(-scale,scale);
      auto b = [&] {return distribution(*rndeng);};
    #if HAVE_CUDA
      cnn::real* t = new cnn::real[val.d.size()];
      generate(t, t + val.d.size(), b);
      CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(cnn::real) * val.d.size(), cudaMemcpyHostToDevice));
      delete[] t;
    #else
      generate(val.v, val.v + val.d.size(), b);
    #endif
    }

    void TensorTools::Randomize(Tensor& d) {
      Randomize(d, sqrt(6) / sqrt(d.d.sum_dims()));
    }

    void TensorTools::RandomBernoulli(Tensor& val, cnn::real p, cnn::real scale) {
      bernoulli_distribution distribution(p);
      auto b = [&] {return distribution(*rndeng) * scale;};
      generate(val.v, val.v + val.d.size(), b);
    }

    void TensorTools::RandomizeNormal(cnn::real mean, cnn::real stddev, Tensor& val) {
      normal_distribution<cnn::real> distribution(mean, stddev);
      auto b = [&] {return distribution(*rndeng);};
      generate(val.v, val.v + val.d.size(), b);
    }
} // namespace cnn
