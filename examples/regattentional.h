#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/rnnem.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "expr-xtra.h"

#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

#define DBG_NEW_RNNEM

namespace cnn {

/// both input and output are continuous-valued vectors 
template <class Builder>
struct RegAttentionalModel{
    explicit RegAttentionalModel(Model& model, 
        unsigned feat_dim, unsigned layers,
        unsigned hidden_dim, unsigned align_dim, bool rnn_src_embeddings,
	    unsigned hidden_replicates=1
    );

    ~RegAttentionalModel(){};

    Expression BuildGraph(const std::vector<vector<cnn::real>> &source,
        const std::vector<vector<cnn::real>>& target, ComputationGraph& cg);

    std::vector<Parameters*> p_h0, p_f0, p_b0;
    Parameters* p_R;
    Parameters* p_P;
    Parameters* p_Q, *p_Qa;
    Parameters* p_bias;
    Parameters* p_Wa;
    Parameters* p_Ua;
    Parameters* p_va;
    Parameters* p_Ta;
    size_t layers; 
    Builder builder;
    Builder builder_src_fwd;
    Builder builder_src_bwd;
    bool rnn_src_embeddings;
    int feat_dim;

    // statefull functions for incrementally creating computation graph, one
    // target word at a time
    void RegAttentionalModel<Builder>::start_new_instance(const std::vector<vector<cnn::real>> &source, ComputationGraph &cg);
    Expression step(int t, ComputationGraph &cg, RNNPointer *prev_state = 0);

    // state variables used in the above two methods
    Expression src;
    Expression i_R;
    Expression i_P;
    Expression i_Q, i_Qa;
    Expression i_bias;
    Expression i_Wa;
    Expression i_Ua;
    Expression i_va;
    Expression i_uax;
    std::vector<Expression> i_h0;
    Expression i_Ta;
    std::vector<Expression> src_fwd, src_bwd;
    unsigned slen;
};

template <class Builder>
RegAttentionalModel<Builder>::RegAttentionalModel(cnn::Model& model,
    unsigned _feat_dim, unsigned layers,
    unsigned hidden_dim, unsigned align_dim, bool _rnn_src_embeddings,
    unsigned hidden_replicates)
    : layers(layers), builder(layers, (_rnn_src_embeddings) ? 2 * hidden_dim : hidden_dim, hidden_dim, &model),
    feat_dim(_feat_dim),
    builder_src_fwd(1, _feat_dim, hidden_dim, &model),
  builder_src_bwd(1, _feat_dim, hidden_dim, &model),
  rnn_src_embeddings(_rnn_src_embeddings)
{
    p_R = model.add_parameters({long(feat_dim), long(hidden_dim)});
    p_bias = model.add_parameters({long(feat_dim)});

    /// for the initial state of the forward and backward source side RNN
    for (auto l = 0; l < hidden_replicates; ++l)
    {
        Parameters *pp = model.add_parameters({ long(hidden_dim) });
        pp->reset_to_zero();
        p_f0.push_back(pp);

        Parameters *pp2 = model.add_parameters({ long(hidden_dim) });
        pp2->reset_to_zero();
        p_b0.push_back(pp2);
    }

    for (auto l = 0; l < hidden_replicates * layers; ++l)
    {
        Parameters *pp = model.add_parameters({ long(hidden_dim) });
        pp->reset_to_zero();
        p_h0.push_back(pp);
    }

    p_Wa = model.add_parameters({ long(align_dim), long(hidden_dim) });
    p_P = model.add_parameters({ long(hidden_dim), long(hidden_dim) });
    if (rnn_src_embeddings) {
        p_Ua = model.add_parameters({ long(align_dim), 2 * long(feat_dim) });
        p_Q = model.add_parameters({ long(2 * hidden_dim), 2*long(feat_dim) });
        p_Qa = model.add_parameters({ long(2 * hidden_dim) });
    }
    else {
        p_Ua = model.add_parameters({ long(align_dim), long(feat_dim) });
        p_Q = model.add_parameters({ long(hidden_dim), long(feat_dim) });
        p_Qa = model.add_parameters({ long(hidden_dim) });
    }
    p_va = model.add_parameters({ long(align_dim) });
}

template <class Builder>
void RegAttentionalModel<Builder>::start_new_instance(const std::vector<vector<cnn::real>> &source, ComputationGraph &cg)
{
    src_fwd.clear();
    src_bwd.clear();
    
    slen = source.size();

    if (!rnn_src_embeddings) {
        src_fwd.resize(slen);
        for (unsigned i = 0; i < slen; ++i)
        {
            long fdim = source[i].size();
            assert(fdim == feat_dim);
            src_fwd[i] = input(cg, { fdim }, &source[i]);
        }
        src = concatenate_cols(src_fwd);
    } else {
	    builder_src_fwd.new_graph(cg);
        i_h0.clear();
        for (const auto &p : p_f0)
            i_h0.push_back(parameter(cg, p));
        builder_src_fwd.start_new_sequence(i_h0);

        builder_src_bwd.new_graph(cg);
        i_h0.clear();
        for (const auto &p : p_b0)
            i_h0.push_back(parameter(cg, p));
        builder_src_bwd.start_new_sequence(i_h0);
        
        src = bidirectional(slen, source, cg, builder_src_fwd, builder_src_bwd, src_fwd, src_bwd); 
    }

    // now for the target sentence
    builder.new_graph(cg); 
    i_h0.clear();
    for (const auto &p: p_h0)
        i_h0.push_back(parameter(cg, p));
    builder.start_new_sequence(i_h0);
    i_R = parameter(cg, p_R); // hidden -> word rep parameter
    i_P = parameter(cg, p_P); // direct from hidden to output
    i_Q = parameter(cg, p_Q); // direct from input to output
    i_Qa = parameter(cg, p_Qa); // direct from input to output
    i_bias = parameter(cg, p_bias);  // word bias
    i_Wa = parameter(cg, p_Wa); 
    i_Ua = parameter(cg, p_Ua);
    i_va = parameter(cg, p_va);
    i_uax = i_Ua * src;
}

template <class Builder>
Expression RegAttentionalModel<Builder>::step(int t, ComputationGraph &cg, RNNPointer *prev_state)
{
    Expression i_r_t;
    try{
        // alignment input -- FIXME: just done for top layer
        auto i_h_tm1 = (t == 0 && i_h0.size() > 0) ? i_h0.back() : builder.final_h().back();

        Expression i_wah = i_Wa * i_h_tm1;

        /// need to do subsampling
        Expression i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));

        Expression i_e_t;
        i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;

        Expression i_alpha_t = softmax(i_e_t);

        Expression input = i_Q * src * i_alpha_t + i_Qa;

        Expression i_y_t;
        if (prev_state)
            i_y_t = builder.add_input(*prev_state, input);
        else
            i_y_t = builder.add_input(input);

        i_r_t = affine_transform({ i_bias, i_R, i_y_t });
    }
    catch (...)
    {
        cerr << "attentional.h :: add_input error " << endl;
        abort();
    }
    return i_r_t;
}

template <class Builder>
Expression RegAttentionalModel<Builder>::BuildGraph(const std::vector<vector<cnn::real>> &source,
    const std::vector<vector<cnn::real>>& target, ComputationGraph& cg)
{
    int nsamples = source.size();

    //std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
    start_new_instance(source, cg);

    vector<Expression> vy(nsamples);
    for (unsigned i = 0; i < nsamples; ++i)
    {
        long fdim = target[i].size();
        assert(fdim == feat_dim);
        vy[i] = input(cg, { fdim }, &target[i]);
    }

    std::vector<Expression> errs;

    for (unsigned t = 0; t < nsamples; ++t) {
        Expression i_y_t = step(t, cg);
        Expression i_m_t = logistic(i_y_t);

        Expression i_r_t = cwise_multiply(src_fwd[t], i_m_t); /// masked noisy spectra
        Expression i_ydist = squared_distance(i_r_t, vy[t]);
        errs.push_back(i_ydist);
    }

    Expression i_nerr = sum(errs) / nsamples;

    return i_nerr;
}

}; // namespace cnn
