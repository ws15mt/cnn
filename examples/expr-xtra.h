#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

using namespace cnn;

// Chris -- this should be a library function
Expression arange(ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<float> *aux_mem) 
{
    aux_mem->clear();
    for (unsigned i = begin; i < end; ++i) 
        aux_mem->push_back((log_transform) ? log(1.0 + i) : i);
    return Expression(&cg, cg.add_input(Dim({(long) (end-begin)}), aux_mem));
}

// Chris -- this should be a library function
Expression repeat(ComputationGraph &cg, unsigned num, float value, std::vector<float> *aux_mem) 
{
    aux_mem->clear();
    aux_mem->resize(num, value);
    return Expression(&cg, cg.add_input(Dim({long(num)}), aux_mem));
}

// Chris -- this should be a library function
Expression dither(ComputationGraph &cg, const Expression &expr, float pad_value, std::vector<float> *aux_mem)
{
    const auto& shape = cg.nodes[expr.i]->dim;
    aux_mem->clear();
    aux_mem->resize(shape.cols(), pad_value);
    Expression padding(&cg, cg.add_input(Dim({shape.cols()}), aux_mem));
    Expression padded = concatenate(std::vector<Expression>({padding, expr, padding}));
    Expression left_shift = pickrange(padded, 2, shape.rows()+2);
    Expression right_shift = pickrange(padded, 0, shape.rows());
    return concatenate_cols(std::vector<Expression>({left_shift, expr, right_shift}));
}

// these expressions can surely be implemented much more efficiently than this
Expression abs(const Expression &expr) 
{
    return rectify(expr) + rectify(-expr); 
}

// binary boolean functions, is it better to use a sigmoid?
Expression eq(const Expression &expr, float value, float epsilon=0.1) 
{
    return min(rectify(expr - (value - epsilon)), rectify(-expr + (value + epsilon))) / epsilon; 
}

Expression geq(const Expression &expr, float value, Expression &one, float epsilon=0.01) 
{
    return min(one, rectify(expr - (value - epsilon)) / epsilon);
        //rectify(1 - rectify(expr - (value - epsilon)));
}

Expression leq(const Expression &expr, float value, Expression &one, float epsilon=0.01) 
{
    return min(one, rectify((value + epsilon) - expr) / epsilon);
    //return rectify(1 - rectify((value + epsilon) - expr));
}

/// do forward and backward embedding
template<class Builder>
Expression bidirectional(int slen, const vector<int>& source, ComputationGraph& cg, LookupParameters* p_cs,
    Builder & encoder_fwd, Builder& encoder_bwd)
{

    std::vector<Expression> source_embeddings;

    std::vector<Expression> src_fwd(slen);
    std::vector<Expression> src_bwd(slen);

    for (int t = 0; t < source.size(); ++t) {
        Expression i_x_t = lookup(cg, p_cs, source[t]);
        src_fwd[t] = encoder_fwd.add_input(i_x_t);
    }
    for (int t = source.size() - 1; t >= 0; --t) {
        Expression i_x_t = lookup(cg, p_cs, source[t]);
        src_bwd[t] = encoder_bwd.add_input(i_x_t);
    }

    for (unsigned i = 0; i < slen - 1; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i + 1] })));
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_fwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

/// do forward and backward embedding on continuous valued vectors
template<class Builder>
Expression bidirectional(int slen, const vector<vector<cnn::real>>& source, ComputationGraph& cg, Builder & encoder_fwd, Builder& encoder_bwd,
    std::vector<Expression>& src_fwd, std::vector<Expression>& src_bwd)
{

    assert(slen == source.size());
    std::vector<Expression> source_embeddings;

    src_fwd.resize(slen);
    src_bwd.resize(slen);

    for (int t = 0; t < source.size(); ++t) {
        long fdim = source[t].size();
        src_fwd[t] = input(cg, { fdim }, &source[t]);
    }
    for (int t = source.size() - 1; t >= 0; --t) {
        long fdim = source[t].size();
        src_bwd[t] = input(cg, { fdim }, &source[t]);
    }

    for (unsigned i = 0; i < slen - 1; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i + 1] })));
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_fwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}
