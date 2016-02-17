#ifndef CNN_GRAD_CHECK_H
#define CNN_GRAD_CHECK_H

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"
#include "cnn/macros.h"
namespace cnn {

class Model;
struct ComputationGraph;

void CheckGrad(Model& m, ComputationGraph& g);

void UnitTest(Expression node, ComputationGraph& g);

} // namespace cnn

#endif
