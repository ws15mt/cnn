#include "cnn/conv.h"

#include <sstream>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <boost/shared_ptr.hpp>

#include "cnn/functors.h"
#if HAVE_CUDA
#include "cnn/cuda.h"
#include "cnn/gpu-ops.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h> 
#include <thrust/transform.h> 
#include <thrust/functional.h>
#endif

using namespace std;

namespace cnn {

string AddVectorToAllColumns::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "fold_rows(" << arg_names[0] << ", " << arg_names[1] << ')';
  return os.str();
}

Dim AddVectorToAllColumns::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2 || xs[0].rows() != xs[1].rows() || xs[0].ndims() != 2 || xs[1].ndims() != 1) {
    cerr << "Bad input dimensions in AddVectorToAllColumns: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in AddVectorToAllColumns");
  }
  return xs[0];
}

void AddVectorToAllColumns::forward(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
    gpu::addVectorToAllColumns(xs[0]->d[0] * xs[0]->d[1], xs[0]->v, xs[1]->d[0], xs[1]->v, fx.v);
#else
  auto y = *fx;
  auto x = **xs[0];
  auto b = **xs[1];
  y = x.colwise() + b.col(0);
#endif
}

void AddVectorToAllColumns::backward(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  assert(i < 2);
#if HAVE_CUDA
  gpu::addVectorToAllColumns_backward(i, dEdf.d.rows(), dEdf.d.cols(), dEdf.v, dEdxi.v);
#else
  if (i == 0) { // x
    (*dEdxi) += (*dEdf);
  } else { // bias
    (*dEdxi).col(0) += (*dEdf).rowwise().sum();
  }
#endif
}

string FoldRows::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "fold_rows(" << arg_names[0] << ", nrows=" << nrows << ')';
  return os.str();
}

Dim FoldRows::dim_forward(const vector<Dim>& xs) const {
  int orows = xs[0].rows() / nrows;
  if ((orows * nrows != xs[0].rows()) || xs.size() != 1 || xs[0].ndims() != 2) {
    cerr << "Bad input dimensions in FoldRows: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in FoldRows");
  }
  return Dim({orows, (long)xs[0].cols()});
}

void FoldRows::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#if HAVE_CUDA
  gpu::foldRows(xs[0]->d.rows(), xs[0]->d.cols(), xs[0]->v, nrows, xs[0]->d.rows() / nrows, fx.v);
#else
  auto x = **xs[0];
  auto y = *fx;
  int orows = y.rows();
  for (int i = 0; i < orows; ++i) {
    for (unsigned j = 0; j < nrows; ++j) {
      if (j)
        y.row(i) += x.row(i * nrows + j);
      else // j = 0
        y.row(i) = x.row(i * nrows);
    }
  }
#endif
}

void FoldRows::backward_impl(const vector<const Tensor*>& xs,
                        const Tensor& fx,
                        const Tensor& dEdf,
                        unsigned i,
                        Tensor& dEdxi) const {
  int orows = fx.d.rows();
  auto d = *dEdf;
  auto di = *dEdxi;
#if HAVE_CUDA
  gpu::foldRows_backward(orows, dEdf.v, dEdxi.d.rows(), dEdxi.d.cols(), dEdxi.v);
#else
  for (int i = 0; i < orows; ++i)
    for (unsigned j = 0; j < nrows; ++j)
      di.row(i * nrows + j) += d.row(i);
#endif
}

string Conv1DNarrow::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "conv1d_narrow(" << arg_names[0] << ", f=" << arg_names[1] << ')';
  return os.str();
}

Dim Conv1DNarrow::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2) {
    cerr << "Conv1DNarrow requires two inputs: " << xs << endl;
    throw std::invalid_argument("Conv1DNarrow requires 2 dimensions");
  }
  int ocols = xs[0].cols() - xs[1].cols() + 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows() ||
      ocols < 1) {
    cerr << "Bad input dimensions in Conv1DNarrow: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in Conv1DNarrow");
  }
  return Dim({(long)xs[0].rows(), ocols});
}

void Conv1DNarrow::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Conv1DNarrow::forward not implemented for CUDA");
#else
    auto x = **xs[0];  // input
      auto f = **xs[1];  // filter
      auto y = *fx;
      const unsigned rows = x.rows();
      const unsigned ycols = dim.cols();
      const unsigned fcols = f.cols();
      for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < ycols; ++j) {
          cnn::real t = 0;
          for (unsigned k = 0; k < fcols; ++k)
            t += f(i, k) * x(i, j + k);
          y(i, j) = t;
        }
      }
#endif
}

void Conv1DNarrow::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
#ifdef HAVE_CUDA
  throw std::runtime_error("Conv1DNarrow::backward not implemented for CUDA");
#else
  // TODO this is a bad implementation- rewrite to use unsupported Eigen tensor library
  assert(i < 2);
  const unsigned rows = xs[0]->d.rows();
  const unsigned ycols = dim.cols();
  const unsigned fcols = xs[1]->d.cols();
  auto d = *dEdf;
  auto di = *dEdxi;
  if (i == 0) { // derivative wrt input x
    auto f = **xs[1];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < ycols; ++j) {
        for (unsigned k = 0; k < fcols; ++k)
          di(i, j + k) += f(i, k) * d(i, j);
      }
    }
  } else { // derivative wrt filter f
    auto x = **xs[0];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < ycols; ++j) {
        for (unsigned k = 0; k < fcols; ++k)
          di(i, k) += x(i, j + k) * d(i, j);
      }
    }
  }
#endif
}

string Conv1DWide::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "conv1d_wide(" << arg_names[0] << ", f=" << arg_names[1] << ')';
  return os.str();
}

Dim Conv1DWide::dim_forward(const vector<Dim>& xs) const {
  if (xs.size() != 2) {
    cerr << "Conv1DWide requires two inputs: " << xs << endl;
    throw std::invalid_argument("Conv1DWide requires two inputs");
  }
  int ocols = xs[0].cols() + xs[1].cols() - 1;
  if (xs[0].ndims() != 2 || xs[1].ndims() != 2 ||
      xs[0].rows() != xs[1].rows()) {
    cerr << "Bad input dimensions in Conv1DWide: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in Conv1DWide");
  }
  return Dim({(long)xs[0].rows(), ocols});
}

void Conv1DWide::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    TensorTools::Zero(fx);
    auto x = **xs[0];  // input
    auto f = **xs[1];  // filter
    auto y = *fx;

    if (xs[1]->d.rows() != xs[0]->d.rows())
    {
        cerr << " filter or kernel needs to have the same row as input" << endl;
        abort();
    }

#if HAVE_CUDA
    const unsigned rows = xs[0]->d.rows();
    const unsigned xcols = xs[0]->d.cols();
    const unsigned fcols = f.cols();

    gpu::conv1dwide(rows, xcols, xs[0]->v, fcols, xs[1]->v, fx.v); 
#else
    const unsigned rows = x.rows();
    const unsigned xcols = x.cols();
    const unsigned fcols = f.cols();

    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < xcols; ++j) {
            const cnn::real xij = x(i, j);
            for (unsigned k = 0; k < fcols; ++k)
                y(i, j + k) += f(i, k) * xij;
        }
    }
#endif
}

void Conv1DWide::backward_impl(const vector<const Tensor*>& xs,
                          const Tensor& fx,
                          const Tensor& dEdf,
                          unsigned i,
                          Tensor& dEdxi) const {
  assert(i < 2);
  auto x = **xs[0];  // input
  auto f = **xs[1];  // filter
  auto y = *fx;

  if (xs[1]->d.rows() != xs[0]->d.rows())
  {
      cerr << " filter or kernel needs to have the same row as input" << endl;
      abort();
  }

#if HAVE_CUDA
  const unsigned rows = xs[0]->d.rows();
  const unsigned xcols = xs[0]->d.cols();
  const unsigned fcols = f.cols();
  gpu::conv1dwide_backward(i, rows, xcols, xs[0]->v, fcols, xs[1]->v, dEdf.v, dEdxi.v);
#else
  const unsigned rows = xs[0]->d.rows();
  const unsigned xcols = xs[0]->d.cols();
  const unsigned fcols = xs[1]->d.cols();
  auto d = *dEdf;
  auto di = *dEdxi;
  if (i == 0) { // derivative wrt input x
    auto f = **xs[1];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < xcols; ++j) {
        for (unsigned k = 0; k < fcols; ++k)
          di(i, j) += f(i, k) * d(i, j + k);
      }
    }
  } else { // derivative wrt filter f
    auto x = **xs[0];
    for (unsigned i = 0; i < rows; ++i) {
      for (unsigned j = 0; j < xcols; ++j) {
        const cnn::real xij = x(i, j);
        for (unsigned k = 0; k < fcols; ++k)
          di(i, k) += xij * d(i, j + k);
      }
    }
  }
#endif
}

string KMaxPooling::as_string(const vector<string>& arg_names) const {
  ostringstream os;
  os << "kmaxpool(" << arg_names[0] << ", k=" << k << ')';
  return os.str();
}

Dim KMaxPooling::dim_forward(const vector<Dim>& xs) const {
  if (k < 1) {
    cerr << "Bad bad k in KMaxPooling: " << k << endl;
    throw std::invalid_argument("bad k in KMaxPooling");
  }
  if (xs[0].ndims() != 2 || (xs[0].cols() < k)) {
    cerr << "Bad input dimensions in KMaxPooling: " << xs << endl;
    throw std::invalid_argument("bad input dimensions in KMaxPooling");
  }
  return Dim({long(xs[0].rows()), long(k)});
}

size_t KMaxPooling::aux_storage_size() const {
  // map of where the entries in f(x) go to entries in x
  return sizeof(int) * dim.size();
}

void KMaxPooling::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  int mi = 0;
#if HAVE_CUDA
  gpu::kMaxPooling(xs[0]->d.rows(), xs[0]->d.cols(), xs[0]->v, k, fx.v, static_cast<int*>(aux_mem));
#else
  auto x = **xs[0];
  auto y=*fx;
  float tmp[1024];
  assert(x.cols() < 1024);
  unsigned mi = 0;
  const unsigned rows = x.rows();
  const unsigned xcols = x.cols();
  int* maxmap = static_cast<int*>(aux_mem);
  for (unsigned i=0; i < rows; ++i) {
    //cerr << "row(" << i << ")=" << x.row(i) << endl;
    for (unsigned j=0; j < xcols; ++j)
      tmp[j] = -x(i,j);
    nth_element(tmp, tmp + (k-1), tmp + xcols);
    const cnn::real c = -tmp[k-1];  // kth largest element in row i
    int tt = 0;
    for (unsigned j = 0; j < xcols; ++j) {
      const cnn::real xij = x(i,j);
      if (xij >= c) {
        //cerr << xij << ' ';
        y(i,tt) = xij;
        //assert(mi < dim.size());
        maxmap[mi++] = j;
        ++tt;
        if (tt == k) break;  // could happen in case of ties
      }
    }
    //cerr << endl; abort();
  }
  assert(mi == dim.size());
#endif
}

void KMaxPooling::backward_impl(const vector<const Tensor*>& xs,
                           const Tensor& fx,
                           const Tensor& dEdf,
                           unsigned i,
                           Tensor& dEdxi) const 
{
  const unsigned rows = dim.rows();
  const unsigned cols = dim.cols();
  const int* maxmap = static_cast<const int*>(aux_mem);
#if HAVE_CUDA
  gpu::kMaxPooling_backward(rows, cols, xs[0]->v, dEdf.d.cols(), dEdf.v, dEdxi.v, maxmap);
#else
  for (unsigned i = 0; i < rows; ++i) {
    int mi = 0;
    for (unsigned j = 0; j < cols; ++j) {
      assert(mi < dim.size());
      const int oj = maxmap[mi++];
      if (oj > (*dEdxi).cols() || oj < 0) {
        cerr << dim << (*fx) << endl << (*dEdxi) << endl;
        cerr << "MM:"; for (int k=0;k < dim.size(); ++k) cerr << ' ' << maxmap[k];
        cerr << endl;
        cerr << "BAD: " << oj << endl; abort();
      }
      (*dEdxi)(i, oj) += (*dEdf)(i, j);
    }
  }
#endif
}


} // namespace cnn
