#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include <algorithm>

using namespace cnn;
using namespace std;

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

/// source [1..T][1..NUTT] is time first and then content from each utterance
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
Expression embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, size_t feat_dim);

vector<size_t> each_sentence_length(const vector<vector<int>>& source);

bool similar_length(const vector<vector<int>>& source);

/// source [1..T][1..NUTT] is time first and then content from each utterance
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
template<class Builder>
Expression bidirectional(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero,
    Builder & encoder_fwd, Builder& encoder_bwd, size_t feat_dim)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings;

    std::vector<Expression> src_fwd(slen);
    std::vector<Expression> src_bwd(slen);

    Expression i_x_t;

    for (int t = 0; t < slen; ++t) {
        vector<Expression> vm;
        for (size_t k = 0; k < nutt; k++)
        {
            if (source[k].size() > t)
                vm.push_back(lookup(cg, p_cs, source[k][t]));
            else
                vm.push_back(input(cg, { (long)feat_dim }, &zero));
        }
        i_x_t = concatenate_cols(vm);
        src_fwd[t] = encoder_fwd.add_input(i_x_t);
    }
    for (int t = slen - 1; t >= 0; --t) {
        vector<Expression> vm;
        for (size_t k = 0; k < nutt; k++)
        {
            if (source[k].size() > t)
                vm.push_back(lookup(cg, p_cs, source[k][t]));
            else
                vm.push_back(input(cg, { (long)feat_dim }, &zero));
        }
        i_x_t = concatenate_cols(vm);
        src_bwd[t] = encoder_bwd.add_input(i_x_t);
    }

    for (unsigned i = 0; i < slen; ++i)
    {
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i] })));
    }

    Expression src = concatenate_cols(source_embeddings);

    return src;
}

template<class Builder>
Expression forward_directional(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero,
    Builder & encoder_fwd, size_t feat_dim)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings;

    std::vector<Expression> src_fwd(slen);

    Expression i_x_t;

    for (int t = 0; t < slen; ++t) {
        vector<Expression> vm;
        for (size_t k = 0; k < nutt; k++)
        {
            if (source[k].size() > t)
                vm.push_back(lookup(cg, p_cs, source[k][t]));
            else
                vm.push_back(input(cg, { (long)feat_dim }, &zero));
        }
        i_x_t = concatenate_cols(vm);
        src_fwd[t] = encoder_fwd.add_input(i_x_t);
    }

    Expression src = concatenate_cols(src_fwd);

    return src;
}

template<class Builder>
Expression backward_directional(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero,
    Builder& encoder_bwd, size_t feat_dim)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings;

    std::vector<Expression> src_bwd(slen);

    Expression i_x_t;

    for (int t = slen - 1; t >= 0; --t) {
        vector<Expression> vm;
        for (size_t k = 0; k < nutt; k++)
        {
            if (source[k].size() > t)
                vm.push_back(lookup(cg, p_cs, source[k][t]));
            else
                vm.push_back(input(cg, { (long)feat_dim }, &zero));
        }
        i_x_t = concatenate_cols(vm);
        src_bwd[t] = encoder_bwd.add_input(i_x_t);
    }

    Expression src = concatenate_cols(src_bwd);

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
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_bwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

vector<Expression> attention_to_source(vector<Expression> & v_src, const vector<size_t>& v_slen,
    Expression i_U, Expression src, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, size_t a_dim, size_t feat_dim, size_t nutt);

vector<Expression> local_attention_to(ComputationGraph& cg, vector<int> v_slen,
    Expression i_Wlp, Expression i_blp, Expression i_vlp,
    Expression i_h_tm1, size_t nutt);

vector<Expression> convert_to_vector(Expression & in, size_t dim, size_t nutt);

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_weight(const vector<size_t>& v_slen, vector<Expression>& src_key, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, size_t a_dim, size_t nutt);

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_to_key_and_retreive_value(vector<Expression> & v_src_val, const vector<size_t>& v_slen,
    const vector<Expression> & i_attention_weight, size_t nutt);
