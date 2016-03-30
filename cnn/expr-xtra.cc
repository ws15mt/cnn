#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/expr-xtra.h"
#include "cnn/data-util.h"

using namespace cnn;
using namespace std;

extern int verbose; 

// Chris -- this should be a library function
Expression arange(ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<cnn::real> *aux_mem) 
{
    aux_mem->clear();
    for (unsigned i = begin; i < end; ++i) 
        aux_mem->push_back((log_transform) ? log(1.0 + i) : i);
    return Expression(&cg, cg.add_input(Dim({ (end-begin)}), aux_mem));
}

// Chris -- this should be a library function
Expression repeat(ComputationGraph &cg, unsigned num, cnn::real value, std::vector<cnn::real> *aux_mem) 
{
    aux_mem->clear();
    aux_mem->resize(num, value);
    return Expression(&cg, cg.add_input(Dim({num}), aux_mem));
}

// Chris -- this should be a library function
Expression dither(ComputationGraph &cg, const Expression &expr, cnn::real pad_value, std::vector<cnn::real> *aux_mem)
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
Expression eq(const Expression &expr, cnn::real value, cnn::real epsilon) 
{
    return min(rectify(expr - (value - epsilon)), rectify(-expr + (value + epsilon))) / epsilon; 
}

Expression geq(const Expression &expr, cnn::real value, Expression &one, cnn::real epsilon) 
{
    return min(one, rectify(expr - (value - epsilon)) / epsilon);
        //rectify(1 - rectify(expr - (value - epsilon)));
}

Expression leq(const Expression &expr, cnn::real value, Expression &one, cnn::real epsilon) 
{
    return min(one, rectify((value + epsilon) - expr) / epsilon);
    //return rectify(1 - rectify((value + epsilon) - expr));
}

/// source [1..T][1..NUTT] is time first and then content from each utterance
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
vector<Expression> embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, unsigned feat_dim)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings;

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
        source_embeddings.push_back(i_x_t);
    }

    return source_embeddings;
}

/// source [1..NUTT][1..T] is speaker first and then content from each utterance
/// return a vector for each time [1..T] and each element has an embedding in matrix for all speakers
vector<Expression> embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = source[0].size();
    for (auto p : source)
    {
        if (slen != p.size())
            runtime_error("input sentences should have same lengths");
    }

    std::vector<Expression> source_embeddings;

    Expression i_x_t;

    for (int t = 0; t < slen; ++t) {
        vector<unsigned> idx; 
        for (size_t k = 0; k < nutt; k++)
            idx.push_back(source[k][t]);
        Expression i_x_t = lookup(cg, p_cs, idx);
        source_embeddings.push_back(i_x_t);
    }

    return source_embeddings;
}

/// organise data in the following format
/// [<first sentence> <second sentence> ...]
/// for example with two sentences with length N1 for the first sentence and length N2 for the second sentence
/// [v_spk1_time0 v_spk1_time1 ... v_spk1_timeN1 | v_spk2_time0 v_spk2_time1 ... v_spk2_timeN2]
vector<Expression> embedding_spkfirst(const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs)
{
    size_t nutt = source.size();

    std::vector<Expression> source_embeddings;

    Expression i_x_t;

    for (size_t k = 0; k < nutt; k++)
    {
        vector<Expression> vm;
        for (int t = 0; t < source[k].size(); ++t) {
            vm.push_back(lookup(cg, p_cs, source[k][t]));
        }
        source_embeddings.push_back(concatenate_cols(vm));
    }

    return source_embeddings;
}

/// organise data in the following format
/// [<first sentence> <second sentence> ...]
/// for example with two sentences with length N1 for the first sentence and length N2 for the second sentence
/// [v_spk1_time0_order1 v_spk1_time0_order2 ... v_spk1_time0_orderN, v_spk1_time1_order1 v_spk1_time1_order2 ... v_spk1_time1_orderN ... v_spk1_timeN1_order1 v_spk1_timeN1_order2 ... v_spk1_timeN1_orderN | v_spk2_time0_order1 v_spk2_time0_order2 ... v_spk2_time0_orderN, v_spk2_time1_order1 v_spk2_time1_order2 ... v_spk2_time1_orderN ... v_spk2_timeN1_order1 v_spk2_timeN1_order2 ... v_spk2_timeN1_orderN]
vector<Expression> hash_embedding_spkfirst(const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, int direct_order, int hash_size)
{
    size_t nutt = source.size();

    std::vector<Expression> source_embeddings;

    Expression i_x_t;

    for (size_t k = 0; k < nutt; k++)
    {
        vector<unsigned int> v_hash = hashing(source[k], direct_order, hash_size);
        vector<Expression> vm;
        for (auto p : v_hash){
            vm.push_back(lookup(cg, p_cs, p));
        }
        source_embeddings.push_back(concatenate_cols(vm));
    }

    return source_embeddings;
}

/// return an expression for the time embedding weight
Expression time_embedding_weight(size_t t, unsigned feat_dim, unsigned slen, ComputationGraph& cg, map<size_t, map<size_t, tExpression>> & m_time_embedding_weight)
{
    if (m_time_embedding_weight.find(t) == m_time_embedding_weight.end() || m_time_embedding_weight[t].find(feat_dim) == m_time_embedding_weight[t].end()
        || m_time_embedding_weight[t][feat_dim].find(slen) == m_time_embedding_weight[t][feat_dim].end()){

        vector<cnn::real> lj(feat_dim, 1 - (t +1) / (slen + 0.0));
        for (size_t k = 0; k < lj.size(); k++)
        {
            lj[k] -= (k + 1.0) / feat_dim * (1 - 2.0 * (t  + 1.0) / slen );
        }
        Expression wgt = input(cg, { feat_dim }, &lj);
        cg.incremental_forward();
        m_time_embedding_weight[t][feat_dim][slen] = wgt;
    }
    return m_time_embedding_weight[t][feat_dim][slen] ;
}

/// representation of a sentence using a single vector
vector<Expression> time_embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, size_t feat_dim, map<size_t, map<size_t, tExpression>>  &m_time_embedding_weight)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings;

    Expression i_x_t;

    for (size_t k = 0; k < nutt; k++)
    {
        vector<Expression> vm;
        int t = 0;
        while (t < source[k].size())
        {
            Expression xij = lookup(cg, p_cs, source[k][t]);
            Expression wgt = time_embedding_weight(t, feat_dim, slen, cg, m_time_embedding_weight); 
            vm.push_back(cwise_multiply(wgt, xij));

            t++;
        }
        i_x_t = sum(vm);
        source_embeddings.push_back(i_x_t);
    }
    return source_embeddings;
}

/// simple average of word embeddings
vector<Expression> average_embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings(nutt);

    Expression i_x_t;

    for (int k = 0; k < nutt; k++)
    {
        vector<Expression> vm;
        int t = 0;
        while (t < source[k].size())
        {
            Expression xij = lookup(cg, p_cs, source[k][t]);
            vm.push_back(xij);

            t++;
        }
        i_x_t = average(vm);
        source_embeddings[k] = i_x_t;
    }
    return source_embeddings;
}

/// simple average of word embeddings
/// vsrc is a vector of expression. 
/// assumes that each expression is a matrix for one sentence. 
/// then the average operation is on each sentence
vector<Expression> average_embedding(const vector<unsigned>& slen, int featdim, const vector<Expression>& vsrc)
{
    size_t nutt = slen.size();

    std::vector<Expression> source_embeddings;

    Expression i_x_t;

    for (int k = 0; k < nutt; k++)
    {
        vector<Expression> vm;
        Expression this_src = reshape(vsrc[k], { slen[k] * featdim });
        int t = 0;
        while (t < slen[k] * featdim )
        {
            Expression xij = pickrange(this_src, t, t + featdim); 
            vm.push_back(xij);
            t+=featdim;
        }
        i_x_t = average(vm);
        source_embeddings.push_back(i_x_t);
    }
    return source_embeddings;
}

vector<unsigned> each_sentence_length(const vector<vector<int>>& source)
{
    /// get each sentence length
    vector<unsigned> slen;
    for (auto p : source)
        slen.push_back(p.size());
    return slen;
}

bool similar_length(const vector<vector<int>>& source)
{
    int imax = -1;
    int imin = 10000;
    /// get each sentence length
    vector<int> slen;
    for (auto p : source)
    {
        imax = std::max<int>(p.size(), imax);
        imin = std::min<int>(p.size(), imin);
    }

    return (fabs((cnn::real)(imax - imin)) < 3.0);
}

/// compute attention weights
vector<Expression> attention_to_source(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression& i_va, // to get attention weight
    Expression& i_Wa, // for target side transformation to alignment space
    Expression& i_h_tm1, 
    Expression& src, 
    unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, cnn::real fscale )
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
        slen += p;

    Expression i_wah = i_Wa * i_h_tm1;  /// [d nutt]
    int stt = 0; 
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wah_each = columnslices(i_wah, a_dim, stt, stt + 1); 
        i_wah_rep.push_back(concatenate_cols(vector<Expression>(v_slen[k], i_wah_each)));   /// [d v_slen[k]]
        stt ++;
    }

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_this_src = columnslices(src, a_dim, istt, istt + v_slen[k]);
        Expression i_e_t = transpose(tanh(i_wah_rep[k] + i_this_src)) * i_va;
        Expression i_input;

        Expression wgt = softmax(fscale * i_e_t);
        v_wgt.push_back(wgt);

        i_input = v_src[k] * wgt;  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
        v_input.push_back(i_input);

        istt += v_slen[k];
    }

    return v_input;
}

/**
compute attention weights using local attention. 
local attention predicts position and does softmax on a [p-D, p+D] window. 
this provides a nice way to trucate variable length input to a fixed-dimension representation and then do softmax. 
use this trick to speed-up training. 

assume that all source sentences have the same length
@parameter v_src: vector of expression. each expression is a source input
@parameter v_slen: vector of int, each element is the length of a source input
@parameter src: a concatenated expression. each element has been transformed to alignement space. 
*/
vector<Expression> attention_to_source_batch(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression& i_va, // to get attention weight
    Expression& i_Wa, // for target side transformation to alignment space
    Expression& i_h_tm1,
    Expression& src,
    unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, cnn::real fscale)
{
    Expression i_c_t;
    int slen = 0;
    unsigned ilen = v_slen[0];
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
    {
        slen += p;
        if (ilen != p)
            runtime_error("attention_to_source_batch: all source sentences need to have the same length");
    }

    vector<Expression> v_attention_to;

    Expression i_wah = i_Wa * i_h_tm1;  /// [d nutt]
    int stt = 0;
    int istt = 0;
    vector<Expression> v_et;

    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wah_each = columnslices(i_wah, a_dim, k, k + 1);
        Expression i_wah_this = concatenate_cols(vector<Expression>(v_slen[k], i_wah_each)); /// [d v_slen[k]]

        Expression i_this_src = columnslices(src, a_dim, istt, istt + v_slen[k]);
        Expression i_e_t = i_this_src + i_wah_this;
        v_et.push_back(i_e_t);
        istt += v_slen[k];
    }

    Expression i_th_t = tanh(concatenate_cols(v_et));
    Expression i_e_t = transpose(i_th_t) * i_va;  /// [sum v_slen[k] , 1]
    Expression i_e_t_matrix = reshape(i_e_t, {ilen, nutt });
    Expression wgt = softmax(fscale * i_e_t_matrix);  /// [ilen, nutt]

    vector<Expression> v_input;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wgt = columnslices(wgt, v_slen[k], k, k + 1); 
        v_wgt.push_back(i_wgt);

        v_input.push_back(v_src[k] * i_wgt); // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
    }

    return v_input;
}

/// use bilinear model for attention
vector<Expression> attention_to_source_bilinear(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_va, Expression i_Wa,
    Expression i_h_tm1, unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, const cnn::real fscale)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
        slen += p;

    Expression i_wa = i_Wa * i_h_tm1 ;  /// [d nutt]
    Expression i_wah;
    if (v_slen.size() > 1)
    {
        i_wah = i_wa + concatenate_cols(vector<Expression>(v_slen.size(), i_va));
    }
    else
        i_wah = i_wa + i_va;
    Expression i_wah_reshaped = reshape(i_wah, { nutt * a_dim });

    Expression i_alpha_t;

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input ;
        Expression i_bilinear = transpose(v_src[k]) * pickrange(i_wah_reshaped, k * a_dim, (k + 1)* a_dim); // [v_slen x 1]
        Expression wgt = softmax(fscale * i_bilinear);
        v_wgt.push_back(wgt);

        i_input = v_src[k] * wgt;  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
        v_input.push_back(i_input);
    }

    return v_input;
}

/** use bilinear model for attention
different from attention_to_source_bilinear
this function doesn't add a bias to input
*/
vector<Expression> attention_using_bilinear(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_Wa, Expression i_h_tm1, unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, Expression& fscale)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
        slen += p;

    Expression i_wa = i_Wa * i_h_tm1;  /// [d nutt]
    Expression i_wah_reshaped = reshape(i_wa, { nutt * a_dim });

    Expression i_alpha_t;

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        Expression i_bilinear = transpose(v_src[k]) * pickrange(i_wah_reshaped, k * a_dim, (k + 1)* a_dim); // [v_slen x 1]
        vector<Expression> vscale(v_slen[k], fscale);
        Expression wgt = softmax(cwise_multiply(concatenate(vscale), i_bilinear));
        v_wgt.push_back(wgt);

        i_input = v_src[k] * wgt;  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
        v_input.push_back(i_input);
    }

    return v_input;
}

/**
use the local attention position prediction, together with the global attention
*/
vector<Expression> attention_using_bilinear_with_local_attention(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_Wa, Expression i_h_tm1, unsigned a_dim, unsigned nutt, vector<Expression>& v_wgt, Expression& fscale,
    vector<Expression>& position)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    vector<Expression> v_position_weight;
    int l = 0;
    for (auto p : v_slen)
    {
        slen += p;
        vector<Expression> pweight;
        for (int k = 0; k < p; k++)
            pweight.push_back(-(k - position[l]) * (k-position[l]) / (50));  /// D = 10, 2*(sigma/2)^2 = 2*100/4 = 50
        l++;
        v_position_weight.push_back(exp(concatenate(pweight)));
    }

    Expression i_wa = tanh(i_Wa * i_h_tm1);  /// [d nutt]
    Expression i_wah_reshaped = reshape(i_wa, { nutt * a_dim });

    Expression i_alpha_t;

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        Expression i_bilinear = transpose(v_src[k]) * pickrange(i_wah_reshaped, k * a_dim, (k + 1)* a_dim); // [v_slen x 1]
        vector<Expression> vscale(v_slen[k], fscale);
//        Expression wgt = softmax(cwise_multiply(concatenate(vscale), i_bilinear));
//        Expression each_utt_wgt = cwise_multiply(wgt, v_position_weight[k]);
//        Expression each_utt_wgt = softmax(cwise_multiply(concatenate(vscale), cwise_multiply(i_bilinear, v_position_weight[k])));
        Expression each_utt_wgt = softmax(cwise_multiply(i_bilinear, v_position_weight[k]));
        v_wgt.push_back(each_utt_wgt);

        i_input = v_src[k] * each_utt_wgt;  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
        v_input.push_back(i_input);
    }

    return v_input;
}

vector<Expression> local_attention_to(ComputationGraph& cg, const vector<unsigned>& v_slen,
    Expression i_Wlp, Expression i_blp, Expression i_vlp,
    Expression i_h_tm1, unsigned nutt)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = v_slen[0];
    vector<Expression> v_attention_to;

    Expression i_wah = i_Wlp * i_h_tm1;
    Expression i_wah_bias = concatenate_cols(vector<Expression>(nutt, i_blp));
    Expression i_position = logistic(i_vlp * tanh(i_wah + i_wah_bias));

    Expression i_position_trans = reshape(i_position, { nutt, 1 });
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_position_each = pick(i_position_trans, k) * (v_slen[k] - 1);

        /// need to do subsampling
        v_attention_to.push_back(i_position_each);
    }
    return v_attention_to;
}


vector<Expression> convert_to_vector(Expression & in, unsigned dim, unsigned nutt)
{
    Expression i_d = reshape(in, { dim * nutt });
    vector<Expression> v_d;

    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_t_kk = pickrange(i_d, k * dim, (k + 1) * dim);
        v_d.push_back(i_t_kk);
    }
    return v_d;
}

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_weight(const vector<unsigned>& v_slen, const Expression& src_key, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, unsigned a_dim, unsigned nutt)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
        slen += p;

    Expression i_wah = i_Wa * i_h_tm1;  /// [d nutt]
    Expression i_wah_reshaped = reshape(i_wah, { nutt * a_dim });

    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wah_each = pickrange(i_wah_reshaped, k * a_dim, (k + 1)*a_dim);  /// [d]
        /// need to do subsampling
        i_wah_rep.push_back(concatenate_cols(std::vector<Expression>(v_slen[k], i_wah_each)));  /// [d v_slen[k]]
    }
    Expression i_wah_m = concatenate_cols(i_wah_rep);  // [d \sum_k v_slen[k]]

    /// compare the input with key for every utterance
    i_e_t = transpose(tanh(i_wah_m + concatenate_cols(vector<Expression>(nutt, src_key)))) * i_va;  // [\sum_k v_slen[k] 1]

    Expression i_alpha_t;

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        int istp = istt + v_slen[k];

        i_input = softmax(pickrange(i_e_t, istt, istp));  // [v_slen[k] 1] 
        v_input.push_back(i_input);

        istt = istp;
    }

    return v_input;
}

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_to_key_and_retreive_value(const Expression& M_t, const vector<unsigned>& v_slen,
    const vector<Expression> & i_attention_weight, unsigned nutt)
{

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        int istp = istt + v_slen[k];

        i_input = M_t * i_attention_weight[k];  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
        v_input.push_back(i_input);

        istt = istp;
    }

    return v_input;
}


Expression bidirectional(int slen, const vector<vector<cnn::real>>& source, ComputationGraph& cg, std::vector<Expression>& src_fwd, std::vector<Expression>& src_bwd)
{

    assert(slen == source.size());
    std::vector<Expression> source_embeddings;

    src_fwd.resize(slen);
    src_bwd.resize(slen);

    for (int t = 0; t < source.size(); ++t) {
        unsigned fdim = source[t].size();
        src_fwd[t] = input(cg, { fdim }, &source[t]);
    }
    for (int t = source.size() - 1; t >= 0; --t) {
        unsigned fdim = source[t].size();
        src_bwd[t] = input(cg, { fdim }, &source[t]);
    }

    for (unsigned i = 0; i < slen - 1; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i + 1] })));
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_bwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

/// returns init hidden for each utt in each layer
vector<vector<Expression>> rnn_h0_for_each_utt(vector<Expression> v_h0, unsigned nutt, unsigned feat_dim)
{
    vector<vector<Expression>> v_each_h0;
    v_each_h0.resize(v_h0.size());
    for (size_t ly = 0; ly < v_h0.size(); ly++)
    {
        Expression i_h = reshape(v_h0[ly], { (nutt * feat_dim) });
        for (size_t k = 0; k < nutt; k++)
        {
            v_each_h0[ly].push_back(pickrange(i_h, k * feat_dim, (k + 1)*feat_dim));
        }
    }

    return v_each_h0;
}

vector<cnn::real> get_value(Expression nd, ComputationGraph& cg)
{
    /// get the top output
    vector<cnn::real> vm;

    vm = as_vector(cg.get_value(nd));

    return vm;
}

vector<cnn::real> get_error(Expression nd, ComputationGraph& cg)
{
    /// get the top output
    vector<cnn::real> vm;

    vm = as_vector(cg.get_error(nd.i));

    return vm;
}


/// return alignment matrix to source
vector<Expression> alignmatrix_to_source(vector<Expression> & v_src, const vector<unsigned>& v_slen,
    Expression i_U, Expression src, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, unsigned a_dim, unsigned feat_dim, unsigned nutt, ComputationGraph& cg)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
        slen += p;

    display_value(i_h_tm1, cg);
    display_value(i_Wa, cg);
    Expression i_wah = i_Wa * i_h_tm1;  /// [d nutt]
    display_value(i_wah, cg);
    Expression i_wah_reshaped = reshape(i_wah, { nutt * a_dim });
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wah_each = pickrange(i_wah_reshaped, k * a_dim, (k + 1)*a_dim);  /// [d]
        /// need to do subsampling
        i_wah_rep.push_back(concatenate_cols(std::vector<Expression>(v_slen[k], i_wah_each)));  /// [d v_slen[k]]
    }
    Expression i_wah_m = concatenate_cols(i_wah_rep);  // [d \sum_k v_slen[k]]
    display_value(i_wah_m, cg);

    display_value(src, cg);
    display_value(i_va, cg);
    i_e_t = transpose(tanh(i_wah_m + src)) * i_va;  // [\sum_k v_slen[k] 1]

    Expression i_alpha_t;

    vector<Expression> v_alignment;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        int istp = istt + v_slen[k];
        display_value(i_e_t, cg);
        Expression i_alignment = softmax(pickrange(i_e_t, istt, istp));
        v_alignment.push_back(i_alignment);

        istt = istp;
    }

    return v_alignment;
}

void display_value(const Expression &source, ComputationGraph &cg, string what_to_say)
{
    const Tensor &a = cg.get_value(source.i);

    cnn::real I = a.d.cols();
    cnn::real J = a.d.rows();

    if (what_to_say.size() > 0)
        cout << what_to_say << endl;
    for (unsigned j = 0; j < J; ++j) {
        for (unsigned i = 0; i < I; ++i) {
            cnn::real v = TensorTools::AccessElement(a, IDX2C(j,i, J)); 
            std::cout << v << ' ';
        }
        std::cout << endl;
    }
}

