#ifndef CNN_RNN_H_
#define CNN_RNN_H_

#include "cnn/cnn.h"
#include "cnn/macros.h"
#include "cnn/rnn-state-machine.h"
#include "cnn/expr.h"
#include <string>

using namespace std;
using namespace cnn::expr;

namespace cnn {

class Model;

BOOST_STRONG_TYPEDEF(int, RNNPointer)
inline void swap(RNNPointer& i1, RNNPointer& i2) {
  RNNPointer t = i1; i1 = i2; i2 = t;
}

// interface for constructing an RNN, LSTM, GRU, etc.
struct RNNBuilder {
  RNNBuilder() : cur(-1) , dparallel (1) {}

  virtual ~RNNBuilder();
  /// for parameter sharing 
  RNNBuilder(const RNNBuilder& ref)
  {
      input_dims = ref.input_dims;
      params = ref.params;
      param_vars = ref.param_vars;
      layers = input_dims.size();
      cur = ref.cur; 
      dparallel = ref.dparallel;
  }

  RNNPointer state() const { return cur; }
  int data_in_parallel() const { return dparallel;  }
  void set_data_in_parallel(int n) { dparallel = n; }

  // call this to reset the builder when you are working with a newly
  // created ComputationGraph object
  void new_graph(ComputationGraph& cg) {
    sm.transition(RNNOp::new_graph);
    new_graph_impl(cg);
  }

  // Reset for new sequence
  // call this before add_input and after new_graph,
  // when starting a new sequence on the same hypergraph.
  // h_0 is used to initialize hidden layers at timestep 0 to given values
  void start_new_sequence(const std::vector<Expression>& h_0={}) {
    sm.transition(RNNOp::start_new_sequence);
    cur = RNNPointer(-1);
    head.clear();
    start_new_sequence_impl(h_0);
  }

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  Expression add_input(const Expression& x) {
    sm.transition(RNNOp::add_input);
    head.push_back(cur);
    int rcp = cur;
    cur = head.size() - 1;
    return add_input_impl(rcp, x);
  }

  // add another timestep by reading in the variable x
  // return the hidden representation of the deepest layer
  Expression add_input(const vector<Expression>& x) {
      sm.transition(RNNOp::add_input);
      head.push_back(cur);
      int rcp = cur;
      cur = head.size() - 1;
      return add_input_impl(rcp, x);
  }

  // add another timestep, but define recurrent connection to prev
  // rather than to head[cur]
  // this can be used to construct trees, implement beam search, etc.
  Expression add_input(const RNNPointer& prev, const Expression& x) {
    sm.transition(RNNOp::add_input);
    head.push_back(prev);
    cur = head.size() - 1;
    return add_input_impl(prev, x);
  }

  // add dependency on an external history
  // this can be used to reset prehistory to null for specific elements 
  // so can run data parallelization on RNNs
  Expression add_input(const std::vector<Expression>& prv_history, const Expression& x) {
      sm.transition(RNNOp::add_input);
      head.push_back(cur);
      cur = head.size() - 1;
      return add_input_impl(prv_history, x);
  }

  // rewind the last timestep - this DOES NOT remove the variables
  // from the computation graph, it just means the next time step will
  // see a different previous state. You can remind as many times as
  // you want.
  void rewind_one_step() {
    cur = head[cur];
  }

  // returns node (index) of most recent output
  virtual Expression back() const = 0;
  // access the final output of each hidden layer
  virtual std::vector<Expression> final_h() const = 0;
  // access the state of each hidden layer, in a format that can be used in
  // start_new_sequence
  virtual std::vector<Expression> final_s() const = 0;
  virtual unsigned num_h0_components() const  = 0;
  // copy the parameters of another builder
  virtual void copy(const RNNBuilder & params) = 0;

  void display(ComputationGraph& cg);

protected:
  virtual void new_graph_impl(ComputationGraph& cg) = 0;
  virtual void start_new_sequence_impl(const std::vector<Expression>& h_0) = 0;
  virtual Expression add_input_impl(int prev, const Expression& x) = 0;
  virtual Expression add_input_impl(int prev, const std::vector<Expression>& x) = 0;  /// each layer has its own input
  virtual Expression add_input_impl(const std::vector<Expression>& prev_history, const Expression& x) = 0;
public:
  /// for parameters
  // first index is layer, then ...
  std::vector<std::vector<Parameters*>> params;
  // first index is layer, then ...
  std::vector<std::vector<Expression>> param_vars;
 public:
  unsigned layers;  /// number of layers
  std::vector<unsigned> input_dims;  /// input dimension at each layer
 private:
  // the state machine ensures that the caller is behaving
  RNNStateMachine sm;
  RNNPointer cur;
  std::vector<RNNPointer> head; // head[i] returns the head position
  int dparallel; /// the number of data points to process in parallel. this is used in the case of loading multiple sentences and process them at the same time
};

struct SimpleRNNBuilder : public RNNBuilder {
  SimpleRNNBuilder() = default;
  explicit SimpleRNNBuilder(unsigned layers,
                            const vector<unsigned>& dims,
                            Model* model,
                            cnn::real i_scale = 1.0,
                            string name = "",
                            bool support_lags=false);
  /// for parameter sharing 
  SimpleRNNBuilder(const SimpleRNNBuilder& ref) 
      : RNNBuilder(ref)
  {
      lagging = ref.lagging;
  }

 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h_0) override;
  Expression add_input_impl(int prev, const Expression& x) override;
  Expression add_input_impl(int prev, const std::vector<Expression>& x) override;
  Expression add_input_impl(const std::vector<Expression>& prev_history, const Expression& x) override;

 public:
  Expression add_auxiliary_input(const Expression& x, const Expression &aux);

  Expression back() const { return h.back().back(); }
  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const { return final_h(); }
  void copy(const RNNBuilder & params) override;

  unsigned num_h0_components() const override { return layers; }

  void set_data_in_parallel(int n);

protected:
    // first index is time, second is layer
  std::vector<std::vector<Expression>> h;

  // initial value of h
  // defaults to zero matrix input
  std::vector<Expression> h0;
  bool lagging;
  std::vector<std::vector<Expression>> biases;
};

struct SimpleRNNBuilderWithELU : public SimpleRNNBuilder {
    SimpleRNNBuilderWithELU() = default;
    explicit SimpleRNNBuilderWithELU(unsigned layers,
        const vector<unsigned>& dims,
        Model* model,
        cnn::real i_scale = 1.0,
        string name = "",
        bool support_lags = false) :SimpleRNNBuilder(layers, dims, model, i_scale, name, support_lags){};
    /// for parameter sharing 
    SimpleRNNBuilderWithELU(const SimpleRNNBuilderWithELU& ref)
        : SimpleRNNBuilder(ref)
    {
    }

protected:
    Expression add_input_impl(int prev, const Expression& x) override;
    Expression add_input_impl(int prev, const std::vector<Expression>& x) override;
    Expression add_input_impl(const std::vector<Expression>& prev_history, const Expression& x) override;

public:
    Expression add_auxiliary_input(const Expression& x, const Expression &aux);
};

} // namespace cnn

#endif
