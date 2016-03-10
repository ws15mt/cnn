#ifndef CNN_EIGEN_TENSOR_H
#define CNN_EIGEN_TENSOR_H

#include <initializer_list>
#include <vector>

#include "cnn/dim.h"
#include "cnn/random.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/macros.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "cnn/cuda.h"
#endif
#include <boost/serialization/array.hpp>

// Following line is commented out because it causes errors with large nets (Antonis)
//#define EIGEN_NO_MALLOC

#include <Eigen/Eigen>

namespace cnn {

#define EIGEN_BACKEND 1

#ifdef USE_DOUBLE
    typedef Eigen::MatrixXd  EMatrix;
    typedef Eigen::VectorXd EVector;
#else
    /// for fast evaluation, use float
    typedef Eigen::MatrixXf  EMatrix;
    typedef Eigen::VectorXf EVector;
#endif

struct Tensor {
  Tensor() = default;
  Tensor(const Dim& d, cnn::real* v) : d(d), v(v) {}
  const Eigen::Map<EMatrix, Eigen::Unaligned> operator*() const {
    assert(d.batch_elems() == 1);
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v, d.rows(), d.cols());
  }
  Eigen::Map<EMatrix, Eigen::Unaligned> operator*() {
    assert(d.batch_elems() == 1);
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v, d.rows(), d.cols());
  }
  // Get the data as a vector
  const Eigen::Map<EVector, Eigen::Unaligned> vec() const {
    return Eigen::Map<EVector, Eigen::Unaligned>(v, d.size());
  }
  Eigen::Map<EVector, Eigen::Unaligned> vec() {
    return Eigen::Map<EVector, Eigen::Unaligned>(v, d.size());
  }
  // Get the pointer for a particular batch, automatically broadcasting if the size is zero
  const cnn::real* batch_ptr(unsigned bid) const {
    assert(d.bd == 1 || bid < d.bd);
    return v + (bid%d.bd)*d.batch_size();
  }
  cnn::real* batch_ptr(unsigned bid) {
    assert(d.bd == 1 || bid < d.bd);
    return v + (bid%d.bd)*d.batch_size();
  }
  // Get the matrix for a particular batch, automatically broadcasting if the size is zero
  const Eigen::Map<EMatrix, Eigen::Unaligned> batch_matrix(unsigned bid) const {
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v + (bid%d.bd)*d.batch_size(), d.rows(), d.cols());
  }
  Eigen::Map<EMatrix, Eigen::Unaligned> batch_matrix(unsigned bid) {
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v + (bid%d.bd)*d.batch_size(), d.rows(), d.cols());
  }
  // Get the data as a matrix, where each "row" is the concatenation of rows and columns,
  // and each "column" is batches
  const Eigen::Map<EMatrix, Eigen::Unaligned> rowcol_matrix() const {
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v, d.rows()*d.cols(), d.batch_elems());
  }
  Eigen::Map<EMatrix, Eigen::Unaligned> rowcol_matrix() {
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v, d.rows()*d.cols(), d.batch_elems());
  }
  // Get the data as a matrix, where each "row" is the concatenation of rows,
  // and each "column" is the concatenation of columns and batches
  const Eigen::Map<EMatrix, Eigen::Unaligned> colbatch_matrix() const {
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v, d.rows(), d.cols()*d.batch_elems());
  }
  Eigen::Map<EMatrix, Eigen::Unaligned> colbatch_matrix() {
    return Eigen::Map<EMatrix, Eigen::Unaligned>(v, d.rows(), d.cols()*d.batch_elems());
  }

  // this is very slow: use sparingly
  inline bool is_valid() const {
#if HAVE_CUDA
    std::cerr << "is_valid() not implemented with HAVE_CUDA\n";
    abort();
#else
    const size_t s = d.size();
    for (unsigned i = 0; i < s; ++i)
      if (std::isnan(v[i]) || std::isinf(v[i])) return false;
    return true;
#endif
  }

  // Get a tensor representing a single batch.
  // If this tensor only has a single batch, then broadcast. Otherwise, check to
  // make sure that the requested batch is smaller than the number of batches.
  // TODO: This is a bit wasteful, as it re-calculates bs.batch_size() every time.
  Tensor batch_elem(unsigned b) const {
    if(d.batch_elems() == 1) {
      return *this;
    } else {
      assert(b < d.batch_elems());
      const unsigned bsize = d.batch_size();
      Dim new_d(d); new_d.bd = 1;
      Tensor ret(new_d, v + bsize * b);
      // std::cerr << "Getting tensor for batch " << (b % d.batch_elems()) << " bsize: " << bsize << ", ptr=" << (long)ret.v << std::endl;
      return ret;
    }
  }

  // get tensors for all batches
  std::vector<Tensor> batch_elems() const {
    if(d.batch_elems() == 1) {
      return std::vector<Tensor>(1, *this); 
    } else {
      std::vector<Tensor> bs(d.batch_elems());
      unsigned bsize = d.batch_size();
      Dim new_d = d; new_d.bd = 1;
      assert (d.batch_elems() >= 0);
      for(unsigned b = 0; b < d.batch_elems(); ++b)
        bs[b] = Tensor(new_d, v + bsize * b);
      return bs;
    }
  }

  Dim d;
  cnn::real* v;
  std::vector<Tensor> bs;

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive& ar, const unsigned int) const {
    ar & d;
#if HAVE_CUDA
    cnn::real * vc = (cnn::real*)malloc(d.size() * sizeof(cnn::real));
    CUDA_CHECK(cudaMemcpy(vc, v, d.size() * sizeof(cnn::real), cudaMemcpyDeviceToHost));
    ar & boost::serialization::make_array(vc, d.size());
    free(vc);
#else
    ar & boost::serialization::make_array(v, d.size());
#endif
  }
  template<class Archive>
  void load(Archive& ar, const unsigned int) {
    ar & d;
#if HAVE_CUDA
    CUDA_CHECK(cudaMalloc(&v, d.size() * sizeof(cnn::real)));
    cnn::real* vc = static_cast<cnn::real*>(std::malloc(d.size() * sizeof(cnn::real)));
    ar & boost::serialization::make_array(vc, d.size());
    CUDA_CHECK(cudaMemcpyAsync(v, vc, d.size() * sizeof(cnn::real), cudaMemcpyHostToDevice));
#else
    v = static_cast<cnn::real*>(cnn_mm_malloc(d.size() * sizeof(cnn::real), sizeof(cnn::real)));
    ar & boost::serialization::make_array(v, d.size());
#endif
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);
cnn::real as_scalar(const Tensor& t);
std::vector<cnn::real> as_vector(const Tensor& v);
std::vector<cnn::real> as_vector(int nsize, const cnn::real* v);

struct TensorTools {
  static void Constant(Tensor& d, cnn::real c);
  static void Zero(Tensor& d);
  static void Randomize(Tensor& val, cnn::real scale);
  static void Randomize(Tensor& d);
  // sample some bernoulli random variables and scale them by scale
  static void RandomBernoulli(Tensor& val, cnn::real p, cnn::real scale = 1.0);
  static void RandomizeNormal(cnn::real mean, cnn::real stddev, Tensor& val);
  // AccessElement and SetElement are very, very slow (potentially) - use appropriately
  static cnn::real AccessElement(const Tensor& v, int index);
  static cnn::real AccessElement(const Tensor& v, const Dim& index);
  static void SetElement(const Tensor& v, int index, cnn::real value);

  static void SetElements(const Tensor& v, const std::vector<cnn::real>& vec);
  static void CopyElements(const Tensor& v, const Tensor& v_src);

  // copy elements in v_src into a memory pointed at v. the size of v is increased by the size of v_src
  static void PushElementsToMemory(int& size, const int buf_size, cnn::real* v, const Tensor& v_src);

};

} // namespace cnn

#endif
