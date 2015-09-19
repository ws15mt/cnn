#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

using namespace cnn;
using namespace std;

typedef vector<cnn::real> FVector;
typedef vector<FVector>   FMatrix;
typedef vector<FMatrix>   FCorpus;
typedef vector<FCorpus*>  FCorpusPointers;

Expression arange(ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<float> *aux_mem);

Expression repeat(ComputationGraph &cg, unsigned num, float value, std::vector<float> *aux_mem);

Expression dither(ComputationGraph &cg, const Expression &expr, float pad_value, std::vector<float> *aux_mem);

// these expressions can surely be implemented much more efficiently than this
Expression abs(const Expression &expr);

// binary boolean functions, is it better to use a sigmoid?
Expression eq(const Expression &expr, float value, float epsilon = 0.1);

Expression geq(const Expression &expr, float value, Expression &one, float epsilon = 0.01);

Expression leq(const Expression &expr, float value, Expression &one, float epsilon = 0.01);

/// do forward and backward embedding
template<class Builder>
Expression bidirectional(int slen, const vector<int>& source, ComputationGraph& cg, LookupParameters* p_cs,
    Builder & encoder_fwd, Builder& encoder_bwd);

/// do forward and backward embedding on continuous valued vectors
template<class Builder>
Expression bidirectional(int slen, const vector<vector<cnn::real>>& source, ComputationGraph& cg, Builder & encoder_fwd, Builder& encoder_bwd,
    std::vector<Expression>& src_fwd, std::vector<Expression>& src_bwd);

/**
usually packs a matrix with real value element
this truncates both source and target 
@mbsize : number of samples
@nutt : number of sentences to process in parallel
*/
vector<vector<Expression>> pack_obs(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg);
