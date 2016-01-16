#ifndef CNN_LSTM_H_
#define CNN_LSTM_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"

using namespace cnn::expr;

namespace cnn {

class Model;

struct LSTMBuilder : public RNNBuilder {
  LSTMBuilder() = default;
  explicit LSTMBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model,
                       cnn::real iscale = 1.0,
                       string name = "");
  LSTMBuilder(const LSTMBuilder& ref)
      : RNNBuilder(ref)
  {}

  Expression back() const { return h.back().back(); }
  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const {
    std::vector<Expression> ret = (c.size() == 0 ? c0 : c.back());
    for(auto my_h : final_h()) ret.push_back(my_h);
    return ret;
  }
  unsigned num_h0_components() const override { return 2 * layers; }
  void copy(const RNNBuilder & params) override;

  void set_data_in_parallel(int n) ;
  std::vector<std::vector<Expression>> biases;

protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(int prev, const Expression& x) override;
  Expression add_input_impl(int prev, const std::vector<Expression>& x) override;
  Expression add_input_impl(const std::vector<Expression>& prv_history, const Expression& x) override;

public:

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
};

} // namespace cnn

#endif
