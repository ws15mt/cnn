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
  cnn::real alpha = GRADIENT_CHECK_PARAM_DELTA; /// change to this alpha, which then shows that the difference between numeric and error propagation is around 10e-5.

  cnn::real E = as_scalar(g.forward());
  cnn::real threshold = pow(10.0, -GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL);

  g.backward();

  bool flag = false;
  const vector<Parameters*>& params = m.parameters_list();
  for (auto pp : params) {
    cerr << "\nPARAMETERS " << pp << " name = " << pp->name << endl;
    Parameters& p = *pp;
    size_t ts = p.dim.size();
    
    size_t sample_step = ts / 10;
    if (ts <= 10) sample_step = 10;
    for (size_t i = 0; i < ts; i += sample_step) {
        cnn::real old, newval;
#if HAVE_CUDA
        cudaMemcpy(&old, &p.values.v[i], sizeof(cnn::real), cudaMemcpyDeviceToHost);
#else
      old = p.values.v[i];
#endif

      newval = old - alpha;
#if HAVE_CUDA
      cudaMemcpy(&p.values.v[i], &newval, sizeof(cnn::real), cudaMemcpyHostToDevice);
#else
      p.values.v[i] = newval;
#endif
      cnn::real E_left = as_scalar(g.forward());

      newval = old + alpha;
#if HAVE_CUDA
      cudaMemcpy(&p.values.v[i] , &newval, sizeof(cnn::real), cudaMemcpyHostToDevice);
#else
      p.values.v[i] = newval;
#endif
      cnn::real E_right = as_scalar(g.forward());
      cnn::real g = (E_right - E_left) / (2*alpha);

      cnn::real grd;
#if HAVE_CUDA
      cudaMemcpy(&grd, &p.g.v[i], sizeof(cnn::real), cudaMemcpyDeviceToHost);
#else
      grd = p.g.v[i];
#endif
      if (g == 0 && grd == 0)
          continue;
      
      cnn::real rel_diff = (g - grd) / (std::max<cnn::real>(fabs(g), fabs(grd)));
      bool wrong = (rel_diff > threshold);
      
      if (wrong)
      {
          flag = true; cerr << "*** relative difference [" << rel_diff * 100 << "%] ";
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
        cnn::real old = v.v[i];
        v.v[i] = old - alpha;
        cnn::real E_left = as_scalar(g.forward());

        v.v[i] = old + alpha;
        cnn::real E_right = as_scalar(g.forward());
        cnn::real g = (E_right - E_left) / (2 * alpha);
        cnn::real f = fabs(g - ag.v[i]);
        cnn::real m = max<cnn::real>(fabs(g), fabs(ag.v[i]));
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
    cnn::real alpha = GRADIENT_CHECK_PARAM_DELTA; /// change to this alpha, which then shows that the difference between numeric and error propagation is around 10e-5.

    VariableIndex iidx = node.i;
    Tensor ov;

    /// do forward pass to have everything initialized
    cnn::real E0 = as_scalar(g.forward());

    /// check only this node
    g.set_last_node_evaluated(iidx);
    cnn::real E = as_scalar(g.incremental_forward());
    ov = g.get_value(node);

    assert(E == E0);

    g.backward();
    vector<cnn::real> grderr = get_error(node, g);

    vector<cnn::real> ivalue = get_value(node, g);

    bool flag = false;
    {
        for (size_t i = 0; i < ivalue.size(); ++i) {
            cnn::real old, newval;
            old = ivalue[i];

            newval = old - alpha;
            Tensor tv = ov;
            tv.v[i] = newval;
            g.set_value(tv, node);
            g.set_last_node_evaluated(iidx);
            cnn::real E_left = as_scalar(g.incremental_forward());

            newval = old + alpha;
            tv = ov;
            tv.v[i] = newval;
            g.set_value(tv, node);
            g.set_last_node_evaluated(iidx);
            cnn::real E_right = as_scalar(g.incremental_forward());
            cnn::real g = (E_right - E_left) / (2 * alpha);

            cnn::real threshold;
            cnn::real grd;
            grd = grderr[i];
            threshold = (cnn::real)pow(10.0,
                max((cnn::real)0.0, ceil(log10(min(fabs(g), fabs(grd))))) - (int)GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL);
            cnn::real diff = fabs(g - grd);
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

