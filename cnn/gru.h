#ifndef CNN_GRU_H_
#define CNN_GRU_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"

namespace cnn {

class Model;

struct GRUBuilder : public RNNBuilder {
  GRUBuilder() = default;
  explicit GRUBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Model* model,
                      float iscale = 1.0);
  GRUBuilder(const GRUBuilder& ref):
      RNNBuilder(ref)
  {}

  void set_data_in_parallel(int n);

  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const { return final_h(); }
  Expression back() const { return h.back().back(); }
  unsigned num_h0_components() const override { return layers; }
  void copy(const RNNBuilder & params) override;

 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;
  Expression add_input_impl(int prev, const Expression& x) override;

  // first index is time, second is layer
  std::vector<std::vector<Expression>> h;

  // initial values of h at each layer
  // - default to zero matrix input
  std::vector<Expression> h0;

public:
  unsigned hidden_dim;
  std::vector<std::vector<Expression>> biases;
};

} // namespace cnn

#endif
