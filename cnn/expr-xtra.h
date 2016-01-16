#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include <algorithm>
#include <map>

using namespace cnn;
using namespace std;

inline bool is_close(cnn::real a, cnn::real b) {
    /// to-do use CNTK's isclose function
    return (fabs(a - b) < 1e-7);
}

Expression arange(ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<cnn::real> *aux_mem);

Expression repeat(ComputationGraph &cg, unsigned num, cnn::real value, std::vector<cnn::real> *aux_mem);

Expression dither(ComputationGraph &cg, const Expression &expr, cnn::real pad_value, std::vector<cnn::real> *aux_mem);

// these expressions can surely be implemented much more efficiently than this
Expression abs(const Expression &expr);

// binary boolean functions, is it better to use a sigmoid?
Expression eq(const Expression &expr, cnn::real value, cnn::real epsilon = 0.1);

Expression geq(const Expression &expr, cnn::real value, Expression &one, cnn::real epsilon = 0.01);

Expression leq(const Expression &expr, cnn::real value, Expression &one, cnn::real epsilon = 0.01);

/// do forward and backward embedding
template<class Builder>
Expression bidirectional(int slen, const vector<int>& source, ComputationGraph& cg, LookupParameters* p_cs,
    Builder & encoder_fwd, Builder& encoder_bwd);

/// source [1..T][1..NUTT] is time first and then content from each utterance
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
vector<Expression> embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, unsigned feat_dim);

/// return an expression for the time embedding weight
typedef std::map<size_t, Expression> tExpression;
Expression time_embedding_weight(size_t t, unsigned feat_dim, unsigned slen, ComputationGraph & cg, map<size_t, map<size_t, tExpression>> & m_time_embedding_weight);

/// following Facebook's MemNN time encoding
/// representation of a sentence using a single vector
vector<Expression> time_embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, size_t feat_dim, map<size_t, map<size_t, tExpression >> &m_time_embedding_weight);

/// simple average of word embeddings
vector<Expression> average_embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs);

vector<unsigned> each_sentence_length(const vector<vector<int>>& source);

bool similar_length(const vector<vector<int>>& source);

/// source [1..T][1..NUTT] is time first and then content from each utterance
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
template<class Builder>
Expression bidirectional(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero,
    Builder & encoder_fwd, Builder &encoder_bwd, size_t feat_dim)
{
    std::vector<Expression> source_embeddings;

    vector<Expression> src_fwd = forward_directional(slen, source, cg, p_cs, zero, encoder_fwd, feat_dim);
    vector<Expression> src_bwd = backward_directional(slen, source, cg, p_cs, zero, encoder_bwd, feat_dim);

    for (unsigned i = 0; i < slen; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

template<class Builder>
Expression bidirectional(unsigned & slen, const vector<int>& source, ComputationGraph& cg, LookupParameters* p_cs, Builder & encoder_fwd, Builder &encoder_bwd)
{
    std::vector<Expression> source_embeddings;

    std::vector<Expression> src_fwd(slen);
    std::vector<Expression> src_bwd(slen);

    Expression i_x_t;

    for (int t = 0; t < slen; ++t) {
        i_x_t = lookup(cg, p_cs, source[t]);
        src_fwd[t] = encoder_fwd.add_input(i_x_t);
    }
    for (int t = slen - 1; t >= 0; --t) {
        i_x_t = lookup(cg, p_cs, source[t]);
        src_bwd[t] = encoder_bwd.add_input(i_x_t);
    }

    for (unsigned i = 0; i < slen; ++i)
    {
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i] })));
    }

    Expression src = concatenate_cols(source_embeddings);

    return src;
}

/// returns init hidden for each utt in each layer
std::vector<std::vector<Expression>> rnn_h0_for_each_utt(std::vector<Expression> v_h0, unsigned nutt, unsigned feat_dim);

/**
data in return has the following format
the index 0,1,2,3,and 4 below are the sentence indices
[0 1 2 x x;
 0 1 2 3 x;
 0 1 2 3 4]
i.e., the redundence is put to the end of a matrix*/
template<class Builder>
std::vector<Expression> forward_directional(unsigned & slen, const std::vector<std::vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, std::vector<cnn::real>& zero,
    Builder & encoder_fwd, unsigned int feat_dim)
{
    unsigned int nutt= source.size();
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
                vm.push_back(input(cg, { feat_dim }, &zero));
        }
        i_x_t = concatenate_cols(vm);
        src_fwd[t] = encoder_fwd.add_input(i_x_t);
    }

    return src_fwd;
}

/**
do backward directional RNN 
the output is a vector of expression.
the element is the top-layer activity
the element of this vector is an expression for that time.
now need to process the data so that the output is
[0 1 2 x x;
 0 1 2 3 x;
 0 1 2 3 4]
i.e., the redundence is put to the end of a matrix*/
template<class Builder>
vector<Expression> backward_directional(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero,
    Builder& encoder_bwd, unsigned int feat_dim)
{
    size_t ly; 
    unsigned int nutt= source.size();
    /// get the maximum length of utternace from all speakers
    vector<int> vlen; 
    bool bsamelength = true; 
    slen = 0;
    for (auto p : source)
    {
        slen = (slen < p.size()) ? p.size() : slen;
        vlen.push_back(p.size());
        if (slen != p.size())
        {
            bsamelength = false;
        }
    }

    std::vector<Expression> source_embeddings;

    std::vector<Expression> src_bwd(slen);

    Expression i_x_t;

    //// the initial hidden state, no data has observed yet
    vector<Expression> v_h0 = encoder_bwd.final_s();
    vector<vector<Expression>> v_each_h0 = rnn_h0_for_each_utt(v_h0, nutt, feat_dim);

    vector<Expression> v_ht = v_h0;
    size_t ik = 0;
    for (int t = slen - 1; t >= 0; --t) {
        vector<vector<Expression>> vhh = rnn_h0_for_each_utt(v_ht, nutt, feat_dim);
        vector<Expression> vm;
        vector<vector<Expression>> vhh_sub = vhh;

        v_ht.clear();

        for (size_t k = 0; k < nutt; k++)
        {
            int j = vlen[k] - t - 1;
            if (j >= 0)
            {
                vm.push_back(lookup(cg, p_cs, source[k][vlen[k] - 1 - j]));
            }
            else
            {
                vm.push_back(input(cg, { feat_dim }, &zero));
                for (ly = 0; ly < v_h0.size(); ly++)
                    vhh_sub[ly][k] = v_each_h0[ly][k];
            }
        }

        for (ly = 0; ly < v_h0.size(); ly++)
        {
            v_ht.push_back(concatenate_cols(vhh_sub[ly]));
        }

        i_x_t = concatenate_cols(vm);
        src_bwd[t] = encoder_bwd.add_input(v_ht, i_x_t);

        v_ht = encoder_bwd.final_s();
        ik++;
    }


    return src_bwd;
}

/// do forward and backward embedding on continuous valued vectors
template<class Builder>
Expression bidirectional(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, Builder * encoder_fwd, Builder* encoder_bwd, unsigned int feat_dim)
{
    std::vector<Expression> source_embeddings;

    vector<Expression> src_fwd = forward_directional(slen, source, cg, p_cs, zero, *encoder_fwd, feat_dim);
    vector<Expression> src_bwd = backward_directional(slen, source, cg, p_cs, zero, *encoder_bwd, feat_dim);

    for (unsigned i = 0; i < slen; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

vector<Expression> convert_to_vector(Expression & in, unsigned dim, unsigned nutt);

vector<Expression> attention_to_source(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_U, Expression src, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, unsigned a_dim, unsigned nutt, vector<Expression>& wgt, cnn::real fscale = 1.0);
vector<Expression> attention_to_source_bilinear(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_va, Expression i_Wa,
    Expression i_h_tm1, unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, cnn::real fscale = 1.0);
vector<Expression> attention_using_bilinear(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_Wa, Expression i_h_tm1, unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, Expression& fscale);
vector<Expression> attention_using_bilinear_with_local_attention(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_Wa, Expression i_h_tm1, unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, Expression& fscale,
    vector<Expression>& position);
vector<Expression> local_attention_to(ComputationGraph& cg, const vector<unsigned> & v_slen,
    Expression i_Wlp, Expression i_blp, Expression i_vlp,
    Expression i_h_tm1, unsigned nutt);

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_weight(const vector<unsigned>& v_slen, const Expression& src_key, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, unsigned a_dim, unsigned nutt);

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_to_key_and_retreive_value(const Expression & M_t, const vector<unsigned>& v_slen,
    const vector<Expression> & i_attention_weight, unsigned nutt);

vector<Expression> alignmatrix_to_source(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_U, Expression src, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, unsigned a_dim, unsigned feat_dim, unsigned nutt, ComputationGraph& cg);
    
vector<cnn::real> get_value(Expression nd, ComputationGraph& cg);
vector<cnn::real> get_error(Expression nd, ComputationGraph& cg);

void display_value(const Expression &source, ComputationGraph &cg, string what_to_say = "");