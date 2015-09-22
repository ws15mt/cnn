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

    for (unsigned i = 0; i < slen-1; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i+1] })));
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_bwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

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
            if (source[k].size() >= t)
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
            if (source[k].size() >= t)
                vm.push_back(lookup(cg, p_cs, source[k][t]));
            else
                vm.push_back(input(cg, { (long)feat_dim }, &zero));
        }
        i_x_t = concatenate_cols(vm);
        src_bwd[t] = encoder_bwd.add_input(i_x_t);
    }

    for (unsigned i = 0; i < slen-1; ++i)
    {
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i+1] })));
    }
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen-1], src_bwd[slen-1] })));

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

    for (unsigned i = 0; i < slen-1; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i+1] })));
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_bwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

template<class Builder>
vector<Expression> attention_to_source(ComputationGraph& cg, vector<int> v_slen,
    Expression i_U, Expression src, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, size_t a_dim, size_t feat_dim,  size_t nutt)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = v_slen[0];
    vector<Expression> i_wah_rep;

    Expression i_wah = i_Wa * i_h_tm1;
    Expression i_wah_reshaped = reshape(i_wah, { long(nutt * a_dim) });
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wah_each = pickrange(i_wah_reshaped, k * a_dim, (k + 1)*a_dim);
        /// need to do subsampling
        i_wah_rep.push_back(concatenate_cols(std::vector<Expression>(v_slen[k], i_wah_each)));
    }
    Expression i_wah_m = concatenate_cols(i_wah_rep);

    i_e_t = transpose(tanh(i_wah_m + i_U * src)) * i_va;

    Expression i_alpha_t;
    Expression src_reshaped = reshape(src, { (long)(feat_dim * slen * nutt) });
    vector<Expression> v_input;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        int istt = k * slen;
        int istp = k * slen + v_slen[k];
        Expression i_src = pickrange(src_reshaped, feat_dim * istt, feat_dim * istp);
        Expression i_src_reshaped = reshape(i_src, { (long)feat_dim, (long)(v_slen[k]) });
        i_input = i_src_reshaped * softmax(pickrange(i_e_t, k * slen, (k + 1) * slen));

        v_input.push_back(i_input);
    }

    return v_input;
}

