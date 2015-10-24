#ifndef CNN_GRAD_CHECK_H
#define CNN_GRAD_CHECK_H

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"

namespace cnn {

    // 10e-5
#define GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL 3
#define GRADIENT_CHECK_PARAM_DELTA 1e-3


class Model;
struct ComputationGraph;

void CheckGrad(Model& m, ComputationGraph& g);

void UnitTest(Expression node, ComputationGraph& g);

} // namespace cnn

#endif
