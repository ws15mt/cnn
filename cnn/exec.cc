#include "cnn/exec.h"

#include "cnn/param-nodes.h"
#include "cnn/expr-xtra.h"

using namespace std;

namespace cnn {

ExecutionEngine::~ExecutionEngine() {}

void SimpleExecutionEngine::invalidate() {
    num_nodes_evaluated = 0;
}

const Tensor& SimpleExecutionEngine::forward() { 
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return forward(node_max_index);
}

const Tensor& SimpleExecutionEngine::forward(VariableIndex i) {
  invalidate();
  return incremental_forward(i);
}

void SimpleExecutionEngine::set_value(const Tensor& t, VariableIndex i) {
    assert(i < cg.nodes.size());
    if (i >= num_nodes_evaluated) {
        cerr << " this is only for adapting parameters. need to precompute node using forward or incremental forward before calling this function" << endl;
        abort();
    }
    nfxs[i] = t;
}

const Tensor& SimpleExecutionEngine::get_value(VariableIndex i) {
    assert(i < cg.nodes.size());
    if (i >= num_nodes_evaluated) {
      incremental_forward();
    }
    return nfxs[i];
}

const Tensor& SimpleExecutionEngine::get_error(VariableIndex i) 
{
    assert(i < cg.nodes.size());
    if (ndEdfs.size() != cg.nodes.size())
    {
        cerr << "need to run backward before calling this function" << endl;
        abort();
    }

    return ndEdfs[i];
}

void SimpleExecutionEngine::set_last_node_evaluated(VariableIndex idx)
{
    num_nodes_evaluated = idx;
}

const Tensor& SimpleExecutionEngine::incremental_forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return incremental_forward(node_max_index);
}

const Tensor& SimpleExecutionEngine::incremental_forward(VariableIndex i) {
  assert(i < cg.nodes.size());

  // free any old memory if this is a new HG
  if (num_nodes_evaluated == 0) fxs->free();

  if (i >= num_nodes_evaluated) {
    nfxs.resize(i + 1);

    //vector<string> dummy(5, "x");
    vector<const Tensor*> xs(16);
    for (; num_nodes_evaluated <= i; ++num_nodes_evaluated) {
      const Node* node = cg.nodes[num_nodes_evaluated];
      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        xs[ai] = &nfxs[arg];
        ++ai;
      }
      nfxs[num_nodes_evaluated].d = node->dim;
      nfxs[num_nodes_evaluated].v = static_cast<cnn::real*>(fxs->allocate(node->dim.size() * sizeof(cnn::real)));
      if (nfxs[num_nodes_evaluated].v == nullptr) {
        cerr << "no more memory space for forward computation. requested " << node->dim.size() << endl;
        cerr << "out of memory\n";
        abort();
      }
      void* aux_mem = nullptr;
      size_t aux_size = node->aux_storage_size();
      if (aux_size) {
        aux_mem = fxs->allocate(aux_size);
        if (!aux_mem) {
            cerr << "no more memory space for auxiliary memory for forward computation. requested " << aux_size << endl;
            cerr << "aux out of memory\n";
            abort();
        }
      }
      node->aux_mem = aux_mem;
      node->forward(xs, nfxs[num_nodes_evaluated]);
    }
  }
  return nfxs[i];
}

void SimpleExecutionEngine::backward(cnn::real * kScalarInit) {
    assert(nfxs.size() == cg.nodes.size());
    backward((VariableIndex)(cg.nodes.size() - 1), kScalarInit);
}

// TODO what is happening with parameter nodes if from_where > param_node_id ?
void SimpleExecutionEngine::backward(VariableIndex from_where, cnn::real * kScalarInit) {
  assert(from_where+1 <= nfxs.size());
  assert(from_where+1 <= cg.nodes.size());
  if (nfxs[from_where].d.size() != 1) {
    cerr << "backward() called on non-scalar node.\n";
    abort();
  }

  const unsigned num_nodes = from_where+1;
  ndEdfs.resize(num_nodes);
  dEdfs->free();
  for (unsigned i = 0; i < num_nodes; ++i) {
    const auto dim = nfxs[i].d;
    ndEdfs[i].d = dim;
    ndEdfs[i].v = static_cast<cnn::real*>(dEdfs->allocate(dim.size() * sizeof(cnn::real)));
    assert(ndEdfs[i].v);
  }
  dEdfs->zero_allocated_memory();
  // initialize dE/dE = 1
  if (kScalarInit == nullptr)
      ndEdfs.back().v = kSCALAR_ONE;
  else
      ndEdfs.back().v = kScalarInit;
  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  vector<bool> needs_derivative(num_nodes, false);
  for (auto i : cg.parameter_nodes)
    needs_derivative[i] = true;

  for (unsigned ni = 0; ni < num_nodes; ++ni) {
    bool nd = needs_derivative[ni];
    for (auto arg : cg.nodes[ni]->args)
      nd |= needs_derivative[arg];
    needs_derivative[ni] = nd;
  }

  // loop in reverse topological order
  vector<const Tensor*> xs;
  for (int i = num_nodes - 1; i >= 0; --i) {
    const Node* node = cg.nodes[i];
    xs.resize(node->arity());
    unsigned ai = 0;
    for (VariableIndex arg : node->args) {
      xs[ai] = &nfxs[arg];
      ++ai;
    }
    ai = 0;
    for (VariableIndex arg : node->args) {
        if (needs_derivative[arg]) {
            node->backward(xs, nfxs[i], ndEdfs[i], ai, ndEdfs[arg]);
      }
      ++ai;
    }
  }

  // accumulate gradients into parameters
  // this is simpler than you might find in some other frameworks
  // since we assume parameters come into the graph as a "function"
  // that returns the current value of the parameters
  for (VariableIndex i : cg.parameter_nodes)
    static_cast<ParameterNodeBase*>(cg.nodes[i])->accumulate_grad(ndEdfs[i]);
}

} // namespace cnn
