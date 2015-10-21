#include "cnn/grad-check.h"

#include <cassert>
#include <iostream>

#include "cnn/model.h"
#include "cnn/cnn.h"
#include "cnn/tensor.h"
#include "cnn/expr.h"
#include "cnn/expr-xtra.h"

using namespace std;

namespace cnn {

void CheckGrad(Model& m, ComputationGraph& g) {
  float alpha = GRADIENT_CHECK_PARAM_DELTA; /// change to this alpha, which then shows that the difference between numeric and error propagation is around 10e-5.

  float E = as_scalar(g.forward());
  g.backward();

  bool flag = false;
  const vector<Parameters*>& params = m.parameters_list();
  for (auto pp : params) {
    cerr << "\nPARAMETERS " << pp << " name = " << pp->name << endl;
    Parameters& p = *pp;
    size_t ts = p.dim.size();
    
    for (size_t i = 0; i < ts; ++i) {
        float old, newval;
#if HAVE_CUDA
        cudaMemcpy(&old, &p.values.v[i], sizeof(float), cudaMemcpyDeviceToHost);
#else
      old = p.values.v[i];
#endif

      newval = old - alpha;
#if HAVE_CUDA
      cudaMemcpy(&p.values.v[i], &newval, sizeof(float), cudaMemcpyHostToDevice);
#else
      p.values.v[i] = newval;
#endif
      float E_left = as_scalar(g.forward());

      newval = old + alpha;
#if HAVE_CUDA
      cudaMemcpy(&p.values.v[i] , &newval, sizeof(float), cudaMemcpyHostToDevice);
#else
      p.values.v[i] = newval;
#endif
      float E_right = as_scalar(g.forward());
      float g = (E_right - E_left) / (2*alpha);

      float threshold;
      float grd;
#if HAVE_CUDA
      cudaMemcpy(&grd, &p.g.v[i], sizeof(float), cudaMemcpyDeviceToHost);
#else
      grd = p.g.v[i];
#endif
      threshold = (float)pow(10.0,
          max((float)0.0, ceil(log10(min(fabs(g), fabs(grd))))) - (int)GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL);
      float diff = fabs(g - grd);
      bool wrong = (std::isnan(diff) || diff > threshold);
      
      if (diff > 0.2)
      {
          cerr << "too large error" << endl;

      }
      if (wrong)
      {
          flag = true; cerr << "***[" << diff << "] ";
          cerr << grd << ' ' << g << endl;
      }
    }
  }

  const vector<LookupParameters*>& lookup_params = m.lookup_parameters_list();
  for (auto pp : lookup_params) {
    cerr << "\nLOOKUP PARAMETERS " << pp << endl;
    LookupParameters& p = *pp;
    size_t ts = p.dim.size();
    for (unsigned j : p.non_zero_grads) {
      cerr << "OBJECT=" << j << endl;
      Tensor& v = p.values[j];
      Tensor& ag = p.grads[j];
      for (size_t i = 0; i < ts; ++i) {
        float old = v.v[i];
        v.v[i] = old - alpha;
        float E_left = as_scalar(g.forward());

        v.v[i] = old + alpha;
        float E_right = as_scalar(g.forward());
        float g = (E_right - E_left) / (2 * alpha);
        float f = fabs(g - ag.v[i]);
        float m = max(fabs(g), fabs(ag.v[i]));
        if (f > 0.1) {
          if (m > 0.f) f /= m;
          if (f > 0.1) { flag = true; cerr << "*** "; }
        }
        if (flag) 
            cerr << ag.v[i] << ' ' << g << endl;
      }
    }
  }

  if (flag) {
    cerr << "\n*** GRADIENT CHECK FAILED ***\n";
  } else {
    cerr << "\nGRADIENT CHECK PASSED\n";
  }
}

void UnitTest(Expression node, ComputationGraph& g) {
    float alpha = GRADIENT_CHECK_PARAM_DELTA; /// change to this alpha, which then shows that the difference between numeric and error propagation is around 10e-5.

    VariableIndex iidx = node.i;
    Tensor ov; 

    /// do forward pass to have everything initialized
    float E0 = as_scalar(g.forward());

    /// check only this node
    g.set_last_node_evaluated( iidx); 
    float E = as_scalar(g.incremental_forward());
    ov = g.get_value(node);

    assert(E == E0);

    g.backward();
    vector<cnn::real> grderr = get_error(node, g);

    vector<cnn::real> ivalue = get_value(node, g);

    bool flag = false;
    {
        for (size_t i = 0; i < ivalue.size(); ++i) {
            float old, newval;
#if HAVE_CUDA
            cudaMemcpy(&old, &p.values.v[i], sizeof(float), cudaMemcpyDeviceToHost);
#else
            old = ivalue[i];
#endif

            newval = old - alpha;
#if HAVE_CUDA
            cudaMemcpy(&p.values.v[i], &newval, sizeof(float), cudaMemcpyHostToDevice);
#else
            Tensor tv = ov;
            tv.v[i] = newval; 
            g.set_value(tv, node);
#endif
            g.set_last_node_evaluated(iidx);
            float E_left = as_scalar(g.incremental_forward());

            newval = old + alpha;
#if HAVE_CUDA
            cudaMemcpy(&p.values.v[i], &newval, sizeof(float), cudaMemcpyHostToDevice);
#else
            tv = ov;
            tv.v[i] = newval;
            g.set_value(tv, node);
#endif
            g.set_last_node_evaluated(iidx);
            float E_right = as_scalar(g.incremental_forward());
            float g = (E_right - E_left) / (2 * alpha);

            float threshold;
            float grd;
#if HAVE_CUDA
            cudaMemcpy(&grd, &p.g.v[i], sizeof(float), cudaMemcpyDeviceToHost);
#else
            grd = grderr[i];
#endif
            threshold = (float)pow(10.0,
                max((float)0.0, ceil(log10(min(fabs(g), fabs(grd))))) - (int)GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL);
            float diff = fabs(g - grd);
            bool wrong = (std::isnan(diff) || diff > threshold);

            if (diff > 0.2)
            {
                cerr << "too large error" << endl;

            }
            if (wrong)
            {
                flag = true; cerr << "***[" << diff << "] ";
                cerr << grd << ' ' << g << endl;
            }
        }
    }

    if (flag) {
        cerr << "\n*** GRADIENT CHECK FAILED ***\n";
    }
    else {
        cerr << "\nGRADIENT CHECK PASSED\n";
    }
}

}

