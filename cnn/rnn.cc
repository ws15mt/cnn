#include "cnn/rnn.h"
#include <boost/lexical_cast.hpp>
#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"
#include "cnn/expr-xtra.h"

using namespace std;
using namespace cnn::expr;
using namespace cnn;

namespace cnn {

enum { X2H=0, H2H, HB, L2H };

RNNBuilder::~RNNBuilder() {}

void RNNBuilder::display(ComputationGraph& cg) {

    for (unsigned i = 0; i < layers; ++i) {
        const std::vector<Expression>& vars = param_vars[i];
        for (size_t i = 0; i < vars.size(); i++)
            display_value(vars[i], cg);
    }
}

SimpleRNNBuilder::SimpleRNNBuilder(unsigned ilayers,
                       const vector<unsigned>& dims,
                       Model* model,
                       cnn::real iscale,
                       string name, 
                       bool support_lags) : lagging(support_lags) 
{
  unsigned input_dim = dims[INPUT_LAYER];
  unsigned hidden_dim = dims[HIDDEN_LAYER];
  string i_name;
  layers = ilayers;
  unsigned layer_input_dim = input_dim;
  input_dims = vector<unsigned>(layers, layer_input_dim);
  
  for (unsigned i = 0; i < layers; ++i) {
    input_dims[i] = layer_input_dim;
    
    if (name.size() > 0)
        i_name = name + "p_x2h" + boost::lexical_cast<string>(i);
    Parameters* p_x2h = model->add_parameters({ hidden_dim, layer_input_dim }, iscale, i_name);
    if (name.size() > 0)
        i_name = name + "p_h2h" + boost::lexical_cast<string>(i);
    Parameters* p_h2h = model->add_parameters({ hidden_dim, hidden_dim }, iscale, i_name);
    if (name.size() > 0)
        i_name = name + "p_hb" + boost::lexical_cast<string>(i);
    Parameters* p_hb = model->add_parameters({ hidden_dim }, iscale, i_name);
    vector<Parameters*> ps = {p_x2h, p_h2h, p_hb};
    if (lagging)
    {
        if (name.size() > 0)
            i_name = name + "lagging" + boost::lexical_cast<string>(i);
        ps.push_back(model->add_parameters({ hidden_dim, hidden_dim }, iscale, i_name));
    }
    params.push_back(ps);
    layer_input_dim = hidden_dim;
  }
}

void SimpleRNNBuilder::new_graph_impl(ComputationGraph& cg) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    Parameters* p_x2h = params[i][X2H];
    Parameters* p_h2h = params[i][H2H];
    Parameters* p_hb = params[i][HB];
    Expression i_x2h =  parameter(cg,p_x2h);
    Expression i_h2h =  parameter(cg,p_h2h);
    Expression i_hb =  parameter(cg,p_hb);
    vector<Expression> vars = {i_x2h, i_h2h, i_hb};

    if (lagging) {
        Parameters* p_l2h = params[i][L2H];
        Expression i_l2h =  parameter(cg,p_l2h);
        vars.push_back(i_l2h);
    }

    param_vars.push_back(vars);
  }
  set_data_in_parallel(data_in_parallel());
}

void SimpleRNNBuilder::set_data_in_parallel(int n)
{
    RNNBuilder::set_data_in_parallel(n);

    biases.clear();
    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];
        Expression bimb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[2]));
        Expression bcmb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[HB]));

        vector<Expression> b = { bimb,
            bcmb
        };
        biases.push_back(b);
    }
}

void SimpleRNNBuilder::start_new_sequence_impl(const vector<Expression>& h_0) {
  h.clear();
  h0 = h_0;
  if (h0.size()) { assert(h0.size() == layers); }
}

Expression SimpleRNNBuilder::add_input_impl(int prev, const Expression &in) {
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));

    Expression x = in;

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];

        Expression y = affine_transform({ biases[i][0], vars[0], x });

        if (prev == -1 && h0.size() > 0)
            y = y + vars[1] * h0[i];
        else if (prev >= 0)
            y = y + vars[1] * h[prev][i];

        x = h[t][i] = tanh(y);
    }
    return h[t].back();
}

Expression SimpleRNNBuilder::add_input_impl(int prev, const std::vector<Expression> &in) {
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));

    assert(in.size() == layers);
    Expression x = in[0];

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];
        Expression x_additional;
        Expression y = affine_transform({ biases[i][0], vars[0], x });

        if (prev == -1 && h0.size() > 0)
            y = y + vars[1] * h0[i];
        else if (prev >= 0)
            y = y + vars[1] * h[prev][i];
        Expression z = tanh(y);
        if (i > 0)
        {
            z = z + in[i];
        }
        x = h[t][i] = z;
    }
    return h[t].back();
}

Expression SimpleRNNBuilder::add_input_impl(const vector<Expression>& prev_history, const Expression &in) {
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));

    Expression x = in;

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];

        //    Expression y = affine_transform({ vars[2], vars[0], x });
        Expression y = affine_transform({ biases[i][0], vars[0], x });

        if (prev_history.size() == layers)
            y = y + vars[1] * prev_history[i];

        x = h[t][i] = tanh(y);
    }
    return h[t].back();
}

Expression SimpleRNNBuilder::add_auxiliary_input(const Expression &in, const Expression &aux) {
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));

  Expression x = in;

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    assert(vars.size() >= L2H + 1);

//    Expression y = affine_transform({ vars[HB], vars[X2H], x, vars[L2H], aux });
    Expression y = affine_transform({ biases[i][1], vars[X2H], x, vars[L2H], aux });

    if (t == 0 && h0.size() > 0)
      y = y + vars[H2H] * h0[i];
    else if (t >= 1)
      y = y + vars[H2H] * h[t-1][i];

    x = h[t][i] = tanh(y);
  }
  return h[t].back();
}

void SimpleRNNBuilder::copy(const RNNBuilder & rnn) {
  const SimpleRNNBuilder & rnn_simple = (const SimpleRNNBuilder&)rnn;
  assert(params.size() == rnn_simple.params.size());
  for(size_t i = 0; i < rnn_simple.params.size(); ++i) {
      params[i][0]->copy(*rnn_simple.params[i][0]);
      params[i][1]->copy(*rnn_simple.params[i][1]);
      params[i][2]->copy(*rnn_simple.params[i][2]);
  }
}

Expression SimpleRNNBuilderWithELU::add_input_impl(int prev, const Expression &in) {
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));

    Expression x = in;

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];

        Expression y = affine_transform({ biases[i][0], vars[0], x });

        if (prev == -1 && h0.size() > 0)
            y = y + vars[1] * h0[i];
        else if (prev >= 0)
            y = y + vars[1] * h[prev][i];

        x = h[t][i] = exponential_linear_units(y);
    }
    return h[t].back();
}

Expression SimpleRNNBuilderWithELU::add_input_impl(int prev, const std::vector<Expression> &in) {
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));

    assert(in.size() == layers);
    Expression x = in[0];

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];
        Expression x_additional;
        Expression y = affine_transform({ biases[i][0], vars[0], x });

        if (prev == -1 && h0.size() > 0)
            y = y + vars[1] * h0[i];
        else if (prev >= 0)
            y = y + vars[1] * h[prev][i];
        Expression z = exponential_linear_units(y);
        if (i > 0)
        {
            z = z + in[i];
        }
        x = h[t][i] = z;
    }
    return h[t].back();
}

Expression SimpleRNNBuilderWithELU::add_input_impl(const vector<Expression>& prev_history, const Expression &in) {
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));

    Expression x = in;

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];

        //    Expression y = affine_transform({ vars[2], vars[0], x });
        Expression y = affine_transform({ biases[i][0], vars[0], x });

        if (prev_history.size() == layers)
            y = y + vars[1] * prev_history[i];

        x = h[t][i] = exponential_linear_units(y);
    }
    return h[t].back();
}

Expression SimpleRNNBuilderWithELU::add_auxiliary_input(const Expression &in, const Expression &aux) {
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));

    Expression x = in;

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];
        assert(vars.size() >= L2H + 1);

        //    Expression y = affine_transform({ vars[HB], vars[X2H], x, vars[L2H], aux });
        Expression y = affine_transform({ biases[i][1], vars[X2H], x, vars[L2H], aux });

        if (t == 0 && h0.size() > 0)
            y = y + vars[H2H] * h0[i];
        else if (t >= 1)
            y = y + vars[H2H] * h[t - 1][i];

        x = h[t][i] = exponential_linear_units(y);
    }
    return h[t].back();
}

} // namespace cnn
