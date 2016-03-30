#include "cnn/param-nodes.h"
#include "cnn/tensor.h"

#include <sstream>

using namespace std;

namespace cnn {

string ConstParameterNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "const_parameters(" << dim << ", " << params << ')';
  return s.str();
}

Dim ConstParameterNode::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

void ConstParameterNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v = params->values.v;
  fx.m_device_id = params->values.m_device_id;
}

void ConstParameterNode::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node: i = " << i << endl;
  abort();
}

string ParameterNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "parameters(" << dim << ", " << params << ')';
  return s.str();
}

Dim ParameterNode::dim_forward(const vector<Dim>& xs) const {
  assert(xs.size() == 0);
  return dim;
}

void ParameterNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  fx.v = params->values.v;
  fx.m_device_id = params->values.m_device_id;
}

void ParameterNode::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node: i = " << i << endl;
  abort();
}

void ParameterNode::accumulate_grad(const Tensor& g) {
  params->accumulate_grad(g);
}

string InputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "constant(" << dim << ')';
  return s.str();
}

Dim InputNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void InputNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
#if HAVE_CUDA
  cudaMemcpyAsync(fx.v, &pdata->front(), dim.size() * sizeof(cnn::real), cudaMemcpyHostToDevice);
#else
  // TODO memcpy is only necessary if pdata->front() points to an unaligned location
  // need to compute this value
  bool is_input_address_aligned = false;
  if (!is_input_address_aligned) {
    memcpy(fx.v, &pdata->front(), dim.size() * sizeof(cnn::real));
  } else {
    fx.v = const_cast<cnn::real*>(&pdata->front());
  }
#endif
  fx.m_device_id = device_id;
}

void InputNode::backward_impl(const vector<const Tensor*>& xs,
                    const Tensor& fx,
                    const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}

string ReferenceNode::as_string(const vector<string>& arg_names) const {
    ostringstream s;
    s << "reference(" << dim << ')';
    return s.str();
}

Dim ReferenceNode::dim_forward(const vector<Dim>& xs) const {
    return dim;
}

void ReferenceNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
    assert(xs.size() == 0);
#if HAVE_CUDA
    cudaMemcpyAsync(fx.v, pdata, dim.size() * sizeof(cnn::real), cudaMemcpyDeviceToDevice);
#else
    // TODO memcpy is only necessary if pdata->front() points to an unaligned location
    // need to compute this value
    bool is_input_address_aligned = false;
    if (!is_input_address_aligned) {
        memcpy(fx.v, pdata, dim.size() * sizeof(cnn::real));
    }
    else {
        fx.v = const_cast<cnn::real*>(pdata);
    }
#endif
    fx.m_device_id = device_id;
}

void ReferenceNode::backward_impl(const vector<const Tensor*>& xs,
    const Tensor& fx,
    const Tensor& dEdf,
    unsigned i,
    Tensor& dEdxi) const {
    cerr << "called backward() on arity 0 node\n";
    abort();
}

string ScalarInputNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "scalar_constant(" << pdata << ')';
  return s.str();
}

Dim ScalarInputNode::dim_forward(const vector<Dim>& xs) const {
  return Dim({1});
}

void ScalarInputNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
#if HAVE_CUDA
  cudaMemcpyAsync(fx.v, pdata, 1 * sizeof(cnn::real), cudaMemcpyHostToDevice);
#else
  fx.v[0] = *pdata;
#endif
  fx.m_device_id = device_id;
}

void ScalarInputNode::backward_impl(const vector<const Tensor*>& xs,
                               const Tensor& fx,
                               const Tensor& dEdf,
                               unsigned i,
                               Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}

string LookupNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "lookup_parameters(|x|=" << params->values.size() << " --> " << dim << ')';
  return s.str();
}

Dim LookupNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

void LookupNode::forward_impl(const vector<const Tensor*>& xs, Tensor& fx) const {
  assert(xs.size() == 0);
  if(pindex) {
    assert(*pindex < params->values.size());
    assert (fx.d.batch_elems() == 1);
#ifdef HAVE_CUDA
    if (params->values[*pindex].m_device_id < 0)
        cudaMemcpyAsync(fx.v, params->values[*pindex].v, fx.d.size() * sizeof(cnn::real), cudaMemcpyHostToDevice);
    else
        fx.v = params->values[*pindex].v;
    fx.m_device_id = device_id;
    if (params->values_for_non_zero_grads.find(*pindex) == params->values_for_non_zero_grads.end())
    {
        cnn::real *v = (cnn::real*) cnn_mm_malloc(sizeof(cnn::real)*fx.d.size(), CNN_ALIGN);
        params->values_for_non_zero_grads[*pindex] = Tensor(fx.d, v, fx.m_device_id); /// working copies for the values
    }
    CUDA_CHECK(cudaMemcpy(params->values_for_non_zero_grads[*pindex].v, fx.v, sizeof(cnn::real)*fx.d.size(), cudaMemcpyDeviceToDevice));   /// have the same value
#else
    fx.v = params->values[*pindex].v;
#endif
  }
  else {
      std::runtime_error("not supported, should be removed"); 
    assert (pindices);
    assert (fx.d.batch_elems() == pindices->size());
#ifdef HAVE_CUDA
    cnn::real *vv = (cnn::real*) cnn_mm_malloc(sizeof(cnn::real)*fx.d.size(), CNN_ALIGN);
#endif
    fx.m_device_id = device_id;
    for (unsigned b = 0; b < pindices->size(); ++b) {
        unsigned i = pindices->at(b);
      assert (i < params->values.size());
      cnn::real* v = fx.v + fx.d.batch_size() * (b % fx.d.batch_elems());
#if HAVE_CUDA
      cudaMemcpyAsync(v, params->values[i].v, fx.d.batch_size() * sizeof(cnn::real), cudaMemcpyHostToDevice);
      params->values_for_non_zero_grads[i] = Tensor({ fx.d.batch_size() }, vv + fx.d.batch_size() * (b % fx.d.batch_elems()), fx.m_device_id); /// working copies for the values
      CUDA_CHECK(cudaMemcpy(vv + fx.d.batch_size() * (b % fx.d.batch_elems()), v, sizeof(cnn::real)*fx.d.batch_size(), cudaMemcpyDeviceToDevice));   /// have the same value
#else
      memcpy(v, params->values[i].v, fx.d.batch_size() * sizeof(cnn::real));
#endif
    }
  }
  fx.m_device_id = device_id;
}

void LookupNode::backward_impl(const vector<const Tensor*>& xs,
                            const Tensor& fx,
                            const Tensor& dEdf,
                            unsigned i,
                            Tensor& dEdxi) const {
  cerr << "called backward() on arity 0 node\n";
  abort();
}

void LookupNode::accumulate_grad(const Tensor& g) {
  if(pindex) {
    params->accumulate_grad(*pindex, g);
  } else {
    assert (pindices);
    const vector<Tensor>& gb = g.batch_elems();
    for (unsigned b = 0; b < pindices->size(); ++b) {
      unsigned i = pindices->at(b);
      assert (i < params->values.size());
      params->accumulate_grad(i, gb[b]);
    }
  }
}

} // namespace cnn
