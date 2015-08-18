#include "cnn/treelstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum { X2I, H2I, C2I, BI, X2F, H2F, C2F, BF, X2O, H2O, C2O, BO, X2C, H2C, BC };

TreeLSTMBuilder::TreeLSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : layers(layers) {
  unsigned layer_input_dim = input_dim;
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameters* p_x2i = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_c2i = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bi = model->add_parameters({hidden_dim});

    // f
    Parameters* p_x2f = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2f = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_c2f = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bf = model->add_parameters({hidden_dim});
    
    // o
    Parameters* p_x2o = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_c2o = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bo = model->add_parameters({hidden_dim});

    // c
    Parameters* p_x2c = model->add_parameters({hidden_dim, layer_input_dim});
    Parameters* p_h2c = model->add_parameters({hidden_dim, hidden_dim});
    Parameters* p_bc = model->add_parameters({hidden_dim});
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameters*> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2f, p_h2f, p_c2f, p_bf, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc};
    params.push_back(ps);
  }  // layers
}

void TreeLSTMBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg, p[X2I]);
    Expression i_h2i = parameter(cg, p[H2I]);
    Expression i_c2i = parameter(cg, p[C2I]);
    Expression i_bi = parameter(cg, p[BI]);
    //f
    Expression i_x2f = parameter(cg, p[X2F]);
    Expression i_h2f = parameter(cg, p[H2F]);
    Expression i_c2f = parameter(cg, p[C2F]);
    Expression i_bf = parameter(cg, p[BF]);
    //o
    Expression i_x2o = parameter(cg, p[X2O]);
    Expression i_h2o = parameter(cg, p[H2O]);
    Expression i_c2o = parameter(cg, p[C2O]);
    Expression i_bo = parameter(cg, p[BO]);
    //c
    Expression i_x2c = parameter(cg, p[X2C]);
    Expression i_h2c = parameter(cg, p[H2C]);
    Expression i_bc = parameter(cg, p[BC]);

    vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2f, i_h2f, i_c2f, i_bf, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);
    
    
  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void TreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

Expression TreeLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    }
    else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }

    // input
    Expression i_ait;
    if (has_prev_state)
      i_ait = affine_transform({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1});
    else
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);

    // forget 
    Expression i_aft;
    if (has_prev_state)
      i_aft = affine_transform({vars[BF], vars[X2F], in, vars[H2F], i_h_tm1, vars[C2F], i_c_tm1});
    else
      i_ait = affine_transform({vars[BF], vars[X2F], in});
    Expression i_ft = logistic(i_aft);

    // write memory cell
    Expression i_awt;
    if (has_prev_state)
      i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);

    // compute new cell value
    if (has_prev_state) {
      Expression i_nwt = cwise_multiply(i_it, i_wt);
      Expression i_crt = cwise_multiply(i_ft, i_c_tm1);
      ct[i] = i_crt + i_nwt;
    }
    else {
      ct[i] = cwise_multiply(i_it, i_wt);
    }
 
    // output
    Expression i_aot;
    if (has_prev_state)
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
      i_aot = affine_transform({vars[BO], vars[X2O], in});
    Expression i_ot = logistic(i_aot);

    // Compute new h value
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cwise_multiply(i_ot, ph_t);
  }
  return ht.back();
}

} // namespace cnn
