#include "cnn/model.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"

#include <unordered_set>
#include <iostream>

#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#if HAVE_CUDA
#include <thrust/version.h>
#include "cnn/gpu-ops.h"
#include "cnn/cuda.h"
#endif

using namespace std;

namespace cnn {

ParametersBase::~ParametersBase() {}

Parameters::Parameters(const Dim& d, cnn::real scale , std::string nodename) : dim(d), name(nodename) {
  values.d = g.d = d;
  values.v = static_cast<cnn::real*>(ps->allocate(d.size() * sizeof(cnn::real)));
  TensorTools::Randomize(values, scale);
  g.v = static_cast<cnn::real*>(ps->allocate(d.size() * sizeof(cnn::real)));

  TensorTools::Zero(g);
}

size_t Parameters::size() const { return dim.size(); }

void Parameters::reset_to_zero()
{
#if HAVE_CUDA
    gpu::set_to_value_of(values.d.size(), values.v, 0.0);
#else
    (*values) *= 0.0;
#endif
}

void Parameters::scale_parameters(cnn::real a) {
  (*g) *= a;
}

void Parameters::squared_l2norm(cnn::real* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(values.d.size(), values.v, sqnorm, true, false);
#else
  *sqnorm = (*values).squaredNorm();
#endif
}

void Parameters::g_squared_l2norm(cnn::real* sqnorm) const {
#if HAVE_CUDA
  gpu::l2_norm_reducer(g.d.size(), g.v, sqnorm, true, false);
#else
  *sqnorm = (*g).squaredNorm();
#endif
}

void Parameters::copy(const Parameters & param) {
  assert(dim == param.dim);
  TensorTools::CopyElements(values, param.values);
  this->name = param.name;
}

void Parameters::accumulate_grad(const Tensor& d) {
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, g.d.size(), kSCALAR_ONE, d.v, 1, g.v, 1));
#else
  *g += *d;
#endif
}

void Parameters::clear() {
  TensorTools::Zero(g);
}

LookupParameters::LookupParameters(unsigned n, const Dim& d, cnn::real scale, std::string nodename) : dim(d), values(n), grads(n), name(nodename) {
  for (unsigned i = 0; i < n; ++i) {
    auto& v = values[i];
    v.d = d;
    v.v = static_cast<cnn::real*>(ps->allocate(d.size() * sizeof(cnn::real)));
    TensorTools::Randomize(v, scale);

    auto& g = grads[i];
    g.d = d;
    g.v = static_cast<cnn::real*>(ps->allocate(d.size() * sizeof(cnn::real)));
    TensorTools::Zero(g);
  }
}

void LookupParameters::scale_parameters(cnn::real a) {
  for (auto& p : values)
    (*p) *= a;
}

void LookupParameters::Initialize(unsigned index, const vector<cnn::real>& val) {
  assert(int(val.size()) == int(dim.size()));
#if HAVE_CUDA
  cerr << "implement LookupParameters::Initialize\n";
  throw cuda_not_implemented("LookupParameters::Initialize");
#else
  memcpy(values[index].v, &val[0], val.size() * sizeof(cnn::real));
#endif
}

size_t LookupParameters::size() const {
  return values.size() * dim.size();
}

void LookupParameters::g_squared_l2norm(cnn::real* sqnorm) const {
#if HAVE_CUDA
  bool acc = false;
  for (auto i : non_zero_grads) {
    gpu::l2_norm_reducer(grads[i].d.size(), grads[i].v, sqnorm, true, acc);
    acc = true;
  }
#else
  real a = 0;
  for (auto i : non_zero_grads)
    a += (*grads[i]).squaredNorm();
  *sqnorm = a;
#endif
}

void LookupParameters::squared_l2norm(cnn::real* sqnorm) const {
#if HAVE_CUDA
  bool acc = false;
  for (unsigned i = 0; i < values.size(); ++i) {
    gpu::l2_norm_reducer(values[i].d.size(), values[i].v, sqnorm, true, acc);
    acc = true;
  }
#else
  cnn::real a = 0;
  for (unsigned i = 0; i < values.size(); ++i)
    a += (*values[i]).squaredNorm();
  *sqnorm = a;
#endif
}

void LookupParameters::copy(const LookupParameters & param) {
  assert(dim == param.dim);
  for(size_t i = 0; i < param.values.size(); ++i)
    TensorTools::CopyElements(values[i], param.values[i]);
  this->name = param.name;
}

void LookupParameters::accumulate_grad(unsigned index, const Tensor& d) {
  non_zero_grads.insert(index);
#if HAVE_CUDA
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, d.d.size(), kSCALAR_ONE, d.v, 1, grads[index].v, 1));
#else
  *grads[index] += *d;
#endif
}

void LookupParameters::clear() {
  for (auto i : non_zero_grads)
    TensorTools::Zero(grads[i]);
  non_zero_grads.clear();
}

Model::~Model() {
  for (auto p : all_params) delete p;
  if (gradient_norm_scratch)
      cnn_mm_free(gradient_norm_scratch); 
}

void Model::project_weights(cnn::real radius) {
  static cnn::real* project_scratch = 0;
  if (!project_scratch)
    project_scratch = (cnn::real*)cnn_mm_malloc(all_params.size() * sizeof(cnn::real), CNN_ALIGN);
  int pi = 0;
  for (auto p : all_params) {
    p->squared_l2norm(&project_scratch[pi]);
    ++pi;
  }
  cnn::real gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += project_scratch[i];
  cerr << "NORM: " << sqrt(gg) << endl;
}

cnn::real Model::gradient_l2_norm() const {
  if (!gradient_norm_scratch)
      gradient_norm_scratch = (cnn::real*)cnn_mm_malloc(all_params.size() * sizeof(cnn::real), CNN_ALIGN);
  int pi = 0;
  for (auto p : all_params) {
    p->g_squared_l2norm(&gradient_norm_scratch[pi]);
    ++pi;
  }
#if HAVE_CUDA
  cnn::real res = 0;
  gpu::l2_norm_reducer(all_params.size(), gradient_norm_scratch, gradient_norm_scratch, false, false);
  cudaMemcpy(&res, gradient_norm_scratch, sizeof(cnn::real),  cudaMemcpyDeviceToHost);
  return sqrt(res);
#else
  cnn::real gg = 0;
  for (int i = 0; i < pi; ++i)
    gg += gradient_norm_scratch[i];
  return sqrt(gg);
#endif
}

Parameters* Model::add_parameters(const Dim& d, cnn::real scale, std::string nodename) {
  Parameters* p = new Parameters(d, scale, nodename);
  all_params.push_back(p);
  params.push_back(p);
  return p;
}

LookupParameters* Model::add_lookup_parameters(unsigned n, const Dim& d, cnn::real scale, std::string nodename) {
  LookupParameters* p = new LookupParameters(n,d, scale, nodename);
  if (nodename != "") p->name = nodename;
  all_params.push_back(p);
  lookup_params.push_back(p);
  return p;
}

void Model::reset_gradient() {
  for (auto p : params) { p->clear(); }
  for (auto p : lookup_params) { p->clear(); }
}

void save_cnn_model(std::string filename, Model* model) {
    std::ofstream out(filename);
    boost::archive::text_oarchive oa(out);
    oa << (*model);
};

void load_cnn_model(std::string filename, Model* model) {
    std::ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> (*model);
};


} // namespace cnn
