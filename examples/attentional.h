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
#include "cnn/expr-xtra.h"

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

template <class Builder>
struct AttentionalModel {
    explicit AttentionalModel(Model& model, 
        unsigned vocab_size_src, unsigned _vocab_size_tgt, unsigned layers,
        unsigned hidden_dim, unsigned align_dim, bool rnn_src_embeddings,
	    bool giza_extensions, unsigned hidden_replicates=1, 
        LookupParameters* cs=0, LookupParameters *ct=0, 
        bool use_external_memory = false , /// this is for extmem rnn memory size
        cnn::real iscale = 1.0  /// random initialization range
    );

    ~AttentionalModel();

    Expression BuildGraph(const std::vector<int> &source, const std::vector<int>& target,
        ComputationGraph& cg, Expression *alignment = 0, bool usePastHitory = false, bool usePastMemory = false);

    Expression BuildGraph(const std::vector<int> &source, const std::vector<int>& target,
        ComputationGraph& cg, vector<Expression> & decoderInit);

    vector<Expression> BuildGraphWithoutNormalization(const std::vector<int> &source,
        const std::vector<int>& target, ComputationGraph& cg, Expression *alignment = nullptr, bool usePastHistory = false, bool usePastMemory = false);

    void display(const std::vector<int> &source, const std::vector<int>& target,
            ComputationGraph& cg, const Expression& alignment, Dict &sd, Dict &td);

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, 
            int beam_width, Dict &tdict);

    std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, Dict &tdict);
    
    std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, Dict &tdict);

    LookupParameters* p_cs;
    LookupParameters* p_ct;
    std::vector<Parameters*> p_h0;
    Parameters* p_R;
    Parameters* p_P;
    Parameters* p_Q;
    Parameters* p_bias;
    //Parameters* p_M;
    Parameters* p_Wa;
    Parameters* p_Ua;
    Parameters* p_va;
    Parameters* p_Ta;
    size_t layers; 
    Builder builder;
    Builder builder_src_fwd;
    Builder builder_src_bwd;
    bool rnn_src_embeddings;
    bool giza_extensions;
    unsigned vocab_size_tgt;

    // statefull functions for incrementally creating computation graph, one
    // target word at a time
    void start_new_instance(const std::vector<int> &src, ComputationGraph &cg);
    void start_new_instance(const std::vector<int> &source, ComputationGraph &cg, vector<Expression>& decoderInit);
    Expression add_input(int tgt_tok, int t, ComputationGraph &cg, RNNPointer *prev_state = 0);
    std::vector<cnn::real> *auxiliary_vector(); // memory management

    // state variables used in the above two methods
    Expression src;
    Expression i_R;
    Expression i_P;
    Expression i_Q;
    Expression i_bias;
    Expression i_Wa;
    Expression i_Ua;
    Expression i_va;
    Expression i_uax;
    std::vector<Expression> i_h0;
    Expression i_Ta;
    Expression i_src_idx;
    Expression i_src_len;
    std::vector<Expression> aligns;
    unsigned slen;
    std::vector<std::vector<cnn::real>*> aux_vecs; // special storage for constant vectors
    unsigned num_aux_vecs;
};

template <class Builder>
AttentionalModel<Builder>::AttentionalModel(cnn::Model& model,
    unsigned vocab_size_src, unsigned _vocab_size_tgt, unsigned layers, unsigned hidden_dim, 
    unsigned align_dim, bool _rnn_src_embeddings, bool _giza_extentions, unsigned hidden_replicates, 
    LookupParameters* cs, LookupParameters *ct, bool use_external_memory = false, cnn::real iscale = 1.0)
    : layers(layers), builder(layers, { (_rnn_src_embeddings) ? 3 * hidden_dim : 2 * hidden_dim, hidden_dim }, &model, iscale),
    builder_src_fwd(1, { hidden_dim, hidden_dim }, &model, iscale),
    builder_src_bwd(1, { hidden_dim, hidden_dim }, &model, iscale),
  rnn_src_embeddings(_rnn_src_embeddings), 
  giza_extensions(_giza_extentions), vocab_size_tgt(_vocab_size_tgt),
  num_aux_vecs(0)
{
    p_cs = (cs) ? cs : model.add_lookup_parameters(vocab_size_src, {hidden_dim}, iscale); 
    p_ct = (ct) ? ct : model.add_lookup_parameters(vocab_size_tgt, {hidden_dim}, iscale); 
    p_R = model.add_parameters({vocab_size_tgt, hidden_dim}, iscale);
    p_bias = model.add_parameters({ vocab_size_tgt }, iscale);

    if (use_external_memory)
    { 
        for (auto l = 0; l < layers; ++l)
        {
            Parameters *pp = model.add_parameters({ RNNEM_MEM_SIZE }, iscale);
            pp->reset_to_zero();
            p_h0.push_back(pp);
        }
#ifdef DBG_NEW_RNNEM
        for (auto l = 0; l < layers; ++l)
        {
            Parameters * pp = model.add_parameters({ hidden_dim, RNNEM_MEM_SIZE }, iscale);
            pp->reset_to_zero(); 
            p_h0.push_back(pp); 
        }
#endif
    }
    for (auto l = 0; l < hidden_replicates * layers; ++l)
    {
        Parameters *pp = model.add_parameters({ hidden_dim }, iscale);
        pp->reset_to_zero();
        p_h0.push_back(pp);
    }

    p_Wa = model.add_parameters({ align_dim, hidden_dim }, iscale);
    p_P = model.add_parameters({ hidden_dim, hidden_dim }, iscale);
    if (rnn_src_embeddings) {
        p_Ua = model.add_parameters({ align_dim, 2 * hidden_dim }, iscale);
        p_Q = model.add_parameters({ hidden_dim, 2 * hidden_dim }, iscale);
    }
    else {
        p_Ua = model.add_parameters({ align_dim, hidden_dim }, iscale);
        p_Q = model.add_parameters({ hidden_dim, hidden_dim }, iscale);
    }
#ifdef ALIGNMENT
    if (giza_extensions) {
        p_Ta = model.add_parameters({align_dim), 9});
    }
#endif
    p_va = model.add_parameters({ align_dim }, iscale);
}

template <class Builder>
AttentionalModel<Builder>::~AttentionalModel()
{
    for (auto v : aux_vecs)
        delete v;
}

template <class Builder>
void AttentionalModel<Builder>::start_new_instance(const std::vector<int> &source, ComputationGraph &cg)
{
    //slen = source.size() - 1; 
    slen = source.size(); 
    if (!rnn_src_embeddings) {
	    std::vector<Expression> source_embeddings;
	    for (unsigned s = 0; s < slen; ++s) 
	        source_embeddings.push_back(lookup(cg, p_cs, source[s]));
	    src = concatenate_cols(source_embeddings); 
    } else {
	    builder_src_fwd.new_graph(cg);
	    builder_src_fwd.start_new_sequence();
	    builder_src_bwd.new_graph(cg);
	    builder_src_bwd.start_new_sequence();
        src = bidirectional(slen, source, cg, p_cs, builder_src_fwd, builder_src_bwd); 
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
    i_bias = parameter(cg, p_bias);  // word bias
    i_Wa = parameter(cg, p_Wa); 
    i_Ua = parameter(cg, p_Ua);
    i_va = parameter(cg, p_va);
    i_uax = i_Ua * src;

#ifdef ALIGNMENT
    if (giza_extensions) {
        i_Ta = parameter(cg, p_Ta);
        i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
        i_src_len = repeat(cg, slen, log(1.0 + slen), auxiliary_vector());
    }

    aligns.clear();
    aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));
#endif
}

template <class Builder>
void AttentionalModel<Builder>::start_new_instance(const std::vector<int> &source, ComputationGraph &cg, vector<Expression>& decoderInit)
{
    //slen = source.size() - 1; 
    slen = source.size();
    if (!rnn_src_embeddings) {
        std::vector<Expression> source_embeddings;
        for (unsigned s = 0; s < slen; ++s)
            source_embeddings.push_back(lookup(cg, p_cs, source[s]));
        src = concatenate_cols(source_embeddings);
    }
    else {
        builder_src_fwd.new_graph(cg);
        builder_src_fwd.start_new_sequence();
        builder_src_bwd.new_graph(cg);
        builder_src_bwd.start_new_sequence();
        src = bidirectional(slen, source, cg, p_cs, builder_src_fwd, builder_src_bwd);
    }

    // now for the target sentence
    builder.new_graph(cg);
    builder.start_new_sequence(decoderInit);
    cg.incremental_forward();
    i_R = parameter(cg, p_R); // hidden -> word rep parameter
    i_P = parameter(cg, p_P); // direct from hidden to output
    i_Q = parameter(cg, p_Q); // direct from input to output
    i_bias = parameter(cg, p_bias);  // word bias
    i_Wa = parameter(cg, p_Wa);
    i_Ua = parameter(cg, p_Ua);
    i_va = parameter(cg, p_va);
    i_uax = i_Ua * src;

#ifdef ALIGNMENT
    if (giza_extensions) {
        i_Ta = parameter(cg, p_Ta);
        i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
        i_src_len = repeat(cg, slen, log(1.0 + slen), auxiliary_vector());
    }

    aligns.clear();
    aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));
#endif
}

template <class Builder>
std::vector<cnn::real>* AttentionalModel<Builder>::auxiliary_vector()
{
    while (num_aux_vecs >= aux_vecs.size())
        aux_vecs.push_back(new std::vector<cnn::real>());
    // NB, we return the last auxiliary vector, AND increment counter
    return aux_vecs[num_aux_vecs++];
}

template <class Builder>
Expression AttentionalModel<Builder>::add_input(int trg_tok, int t, ComputationGraph &cg, RNNPointer *prev_state)
{
    Expression i_r_t;
    try{
        // alignment input -- FIXME: just done for top layer
        auto i_h_tm1 = (t == 0 && i_h0.size() > 0) ? i_h0.back() : builder.final_h().back();
        if (verbose)
            display_value(i_h_tm1, cg);
        //Expression i_e_t = tanh(i_src_M * i_h_tm1); 
        Expression i_wah = i_Wa * i_h_tm1;
        if (verbose)
            display_value(i_wah, cg);
        // want numpy style broadcasting, but have to do this manually
        Expression i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
        if (verbose)
            display_value(i_wah_rep, cg);
        Expression i_e_t;
        if (giza_extensions) {
#ifdef ALIGNMENT
            std::vector<Expression> alignment_context;
            if (t >= 1) {
                auto i_aprev = concatenate_cols(aligns);
                auto i_asum = sum_cols(i_aprev);
                auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());
                alignment_context.push_back(i_asum_pm);
                auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
                alignment_context.push_back(i_alast_pm);
            }
            else {
                // just 6 repeats of the 0 vector
                auto zeros = repeat(cg, slen, 0, auxiliary_vector());
                //WTF(zeros);
                alignment_context.push_back(zeros); 
                alignment_context.push_back(zeros);
                alignment_context.push_back(zeros);
                alignment_context.push_back(zeros);
                alignment_context.push_back(zeros);
                alignment_context.push_back(zeros);
            }
            //WTF(i_src_idx);
            alignment_context.push_back(i_src_idx);
            //WTF(i_src_len);
            alignment_context.push_back(i_src_len);
            auto i_tgt_idx = repeat(cg, slen, log(1.0 + t), auxiliary_vector());
            //WTF(i_tgt_idx);
            alignment_context.push_back(i_tgt_idx);
            auto i_context = concatenate_cols(alignment_context);
            //WTF(i_context);

            auto i_e_t_input = i_wah_rep + i_uax + i_Ta * transpose(i_context); 
            //WTF(i_e_t_input);
            i_e_t = transpose(tanh(i_e_t_input)) * i_va;
            //WTF(i_e_t);
#endif
        }
        else {
            i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
            if (verbose)
                display_value(i_e_t, cg);
        }
        Expression i_alpha_t = softmax(i_e_t);
        if (verbose)
            display_value(i_alpha_t, cg);
#ifdef ALIGNMENT
        aligns.push_back(i_alpha_t);
#endif
        Expression i_c_t = src * i_alpha_t;
        if (verbose)
            display_value(i_c_t, cg);
        // word input
        Expression i_x_t = lookup(cg, p_ct, trg_tok);
        if (verbose)
            display_value(i_x_t, cg);
        Expression input = concatenate(std::vector<Expression>({ i_x_t, i_c_t })); // vstack/hstack?
        if (verbose)
            display_value(input, cg);
        // y_t = RNN([x_t, a_t])
        Expression i_y_t;
        if (prev_state)
            i_y_t = builder.add_input(*prev_state, input);
        else
            i_y_t = builder.add_input(input);
        if (verbose)
            display_value(i_y_t, cg);

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
Expression AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
    const std::vector<int>& target, ComputationGraph& cg, Expression *alignment, bool usePastHistory, bool usePastMemory)
{
    if (usePastHistory == false)
    {
        for (auto pp = p_h0.begin(); pp != p_h0.end(); pp++)
        {
            (*pp)->reset_to_zero();
        }
    }
    if (usePastMemory)
    {
        size_t i = 0;
        for (auto pp = p_h0.begin(); pp != p_h0.end(); pp++, i++)
        {
            if (i < layers || i >= 2 * layers)
                (*pp)->reset_to_zero();
        }
    }
    //std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
    start_new_instance(source, cg);

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1;
    for (unsigned t = 0; t < tlen; ++t) {
        Expression i_r_t = add_input(target[t], t, cg);
        Expression i_e_t = log_softmax(i_r_t);
        errs.push_back(-pick(i_e_t, target[t + 1]));
        if (verbose)
            display_value(errs.back(), cg);
    }

#ifdef ALIGNMENT
    // save the alignment for later
    if (alignment != 0) {
        // pop off the last alignment column
        *alignment = concatenate_cols(aligns);
    }
#endif
    Expression i_nerr = sum(errs);

    return i_nerr;
}

/// build graph with decoder initial state comming from other Expression
template <class Builder>
Expression AttentionalModel<Builder>::BuildGraph(const std::vector<int> &source,
    const std::vector<int>& target, ComputationGraph& cg, vector<Expression>& decoderInit)
{

    //std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
    start_new_instance(source, cg, decoderInit);

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1;
    for (unsigned t = 0; t < tlen; ++t) {
        Expression i_r_t = add_input(target[t], t, cg);
        errs.push_back(pickneglogsoftmax(i_r_t, target[t + 1]));
        if (verbose)
            display_value(errs.back(), cg);
    }

#ifdef ALIGNMENT
    // save the alignment for later
    if (alignment != 0) {
        // pop off the last alignment column
        *alignment = concatenate_cols(aligns);
    }
#endif
    Expression i_nerr = sum(errs);

    return i_nerr;
}

/// the scores before softmax normalization so that
/// attention model output can be used for combination with other models
template <class Builder>
vector<Expression> AttentionalModel<Builder>::BuildGraphWithoutNormalization(const std::vector<int> &source,
    const std::vector<int>& target, ComputationGraph& cg, Expression *alignment, bool usePastHistory, bool usePastMemory)
{
    if (usePastHistory == false)
    {
        for (auto pp = p_h0.begin(); pp != p_h0.end(); pp++)
        {
            (*pp)->reset_to_zero();
        }
    }
    if (usePastMemory)
    {
        size_t i = 0;
        for (auto pp = p_h0.begin(); pp != p_h0.end(); pp++, i++)
        {
            if (i < layers || i >= 2 * layers)
                (*pp)->reset_to_zero();
        }
    }
    //std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;
    start_new_instance(source, cg);

    std::vector<Expression> errs;
    const unsigned tlen = target.size() - 1;
    for (unsigned t = 0; t < tlen; ++t) {
        Expression i_r_t = add_input(target[t], t, cg);
        errs.push_back(i_r_t);
    }
#ifdef ALIGNMENT
    // save the alignment for later
    if (alignment != 0) {
        // pop off the last alignment column
        *alignment = concatenate_cols(aligns);
    }
#endif
    return errs;
}

template <class Builder>
void 
AttentionalModel<Builder>::display(const std::vector<int> &source, const std::vector<int>& target, 
                          ComputationGraph &cg, const Expression &alignment, Dict &sd, Dict &td)
{
    using namespace std;

    // display the alignment
    //cnn::real I = target.size() - 1;
    //cnn::real J = source.size() - 1;
    cnn::real I = target.size();
    cnn::real J = source.size();
    //vector<string> symbols{"\u2588","\u2589","\u258A","\u258B","\u258C","\u258D","\u258E","\u258F"};
    vector<string> symbols{".","o","*","O","@"};
    int num_symbols = symbols.size();
    vector<cnn::real> thresholds;
    thresholds.push_back(0.8/I);
    cnn::real lgap = (0 - log(thresholds.back())) / (num_symbols - 1);
    for (auto rit = symbols.begin(); rit != symbols.end(); ++rit) {
        cnn::real thr = exp(log(thresholds.back()) + lgap);
        thresholds.push_back(thr);
    }
    // FIXME: thresholds > 1, what's going on?
    //cout << thresholds.back() << endl;

    const Tensor &a = cg.get_value(alignment.i);
    //WTF(alignment);
    //cout << "I = " << I << " J = " << J << endl;

    cout.setf(ios_base::adjustfield, ios_base::left);
    cout << setw(12) << "source" << "  ";
    cout.setf(ios_base::adjustfield, ios_base::right);
    for (int j = 0; j < J; ++j) 
        cout << setw(2) << j << ' ';
    cout << endl;

    for (unsigned i = 0; i < I; ++i) {
        cout.setf(ios_base::adjustfield, ios_base::left);
        //cout << setw(12) << td.Convert(target[i+1]) << "  ";
        cout << setw(12) << td.Convert(target[i]) << "  ";
        cout.setf(ios_base::adjustfield, ios_base::right);
        cnn::real max_v = 0;
        int max_j = -1;
        for (unsigned j = 0; j < J; ++j) {
            cnn::real v = TensorTools::AccessElement(a, { j, i });
            string symbol;
            for (int s = 0; s <= num_symbols; ++s) {
                if (s == 0) 
                    symbol = ' ';
                else
                    symbol = symbols[s-1];
                if (s != num_symbols && v < thresholds[s])
                    break;
            }
            cout << setw(2) << symbol << ' ';
            if (v >= max_v) {
                max_v = v;
                max_j = j;
            }
        }
        cout << setw(20) << "max Pr=" << setprecision(3) << setw(5) << max_v << " @ " << max_j << endl;
    }
    cout << resetiosflags(ios_base::adjustfield);
    for (int j = 0; j < J; ++j) 
        cout << j << ":" << sd.Convert(source[j]) << ' ';
    cout << endl;
}

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
{
    assert(beam_width == 1); // beam search not implemented 
    const int sos_sym = tdict.Convert("<s>");
    const int eos_sym = tdict.Convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym); 

    std::cerr << tdict.Convert(target.back());
    int t = 0;
    start_new_instance(source, cg);
    while (target.back() != eos_sym) 
    {
        Expression i_scores = add_input(target.back(), t, cg);
        Expression ydist = softmax(i_scores); // compiler warning, but see below

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
        }

        // break potential infinite loop
        if (t > 100) {
            w = eos_sym;
            pr_w = dist[w];
        }

        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
        t += 1;
        target.push_back(w);
    }
    std::cerr << std::endl;

    return target;
}

struct Hypothesis {
    Hypothesis(RNNPointer state, int tgt, cnn::real cst, int _t)
        : builder_state(state), target({tgt}), cost(cst), t(_t) {}
    Hypothesis(RNNPointer state, int tgt, cnn::real cst, Hypothesis &last)
        : builder_state(state), target(last.target), cost(cst), t(last.t+1) {
        target.push_back(tgt);
    }
    RNNPointer builder_state;
    std::vector<int> target;
    cnn::real cost;
    int t;
};

struct CompareHypothesis
{
    bool operator()(const Hypothesis& h1, const Hypothesis& h2)
    {
        if (h1.cost < h2.cost) return true;
        return false; 
    }
};

template <class Builder>
std::vector<int> 
AttentionalModel<Builder>::beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, 
        cnn::Dict &tdict)
{
    assert(!giza_extensions);
    const int sos_sym = tdict.Convert("<s>");
    const int eos_sym = tdict.Convert("</s>");

    size_t tgt_len = 2 * source.size();

    start_new_instance(source, cg);

    priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> completed;
    priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> chart;
    chart.push(Hypothesis(builder.state(), sos_sym, 0.0f, 0));

    boost::integer_range<int> vocab = boost::irange(0, (int)vocab_size_tgt);
    vector<int> vec_vocab(vocab_size_tgt, 0);
    for (auto k : vocab)
    {
        vec_vocab[k] = k;
    }
    vector<int> org_vec_vocab = vec_vocab;

    size_t it = 0;
    while (it < tgt_len) {
        priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> new_chart;
        vec_vocab = org_vec_vocab;
        real best_score = -numeric_limits<real>::infinity() + 100.;

        while(!chart.empty()) {
            Hypothesis hprev = chart.top();
            Expression i_scores = add_input(hprev.target.back(), hprev.t, cg, &hprev.builder_state);
            Expression ydist = softmax(i_scores); // compiler warning, but see below

            // find the top k best next words
            unsigned w = 0;
            auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
            real mscore = log(*max_element(dist.begin(), dist.end())) + hprev.cost; 
            if (mscore < best_score - beam_width)
            {
                chart.pop();
                continue;
            }

            best_score = max(mscore, best_score);

            // add to chart
            size_t k = 0;
            for (auto vi : vec_vocab){
                real score = hprev.cost + log(dist[vi]);
                if (score >= best_score - beam_width)
                {
                    Hypothesis hnew(builder.state(), vi, score, hprev);
                    if (vi == eos_sym)
                        completed.push(hnew);
                    else
                        new_chart.push(hnew);
                }
            }

            chart.pop();
        }

        if (new_chart.size() == 0)
            break;

        // beam pruning
        while (!new_chart.empty())
        {
            if (new_chart.top().cost > best_score - beam_width){
                chart.push(new_chart.top());
            }
            else
                break;
            new_chart.pop();
        }
        it++;
    }

    vector<int> best;
    if (completed.size() == 0)
    {
        cerr << "beam search decoding beam width too small, use the best path so far" << flush;

        best = chart.top().target;
        best.push_back(eos_sym);
    }
    else
        best = completed.top().target;

    for (auto p : best)
    {
        std::cerr << " " << tdict.Convert(p) << " ";
    }
    cerr << endl; 

    return best;
}

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict &tdict)
{
    const int sos_sym = tdict.Convert("<s>");
    const int eos_sym = tdict.Convert("</s>");

    std::vector<int> target;
    target.push_back(sos_sym); 

    std::cerr << tdict.Convert(target.back());
    int t = 0;
    start_new_instance(source, cg);
    while (target.back() != eos_sym) 
    {
        Expression i_scores = add_input(target.back(), t, cg);
        Expression ydist = softmax(i_scores);

	// in rnnlm.cc there's a loop around this block -- why? can incremental_forward fail?
        auto dist = as_vector(cg.incremental_forward());
	cnn::real p = rand01();
        unsigned w = 0;
        for (; w < dist.size(); ++w) {
	    p -= dist[w];
	    if (p < 0) break;
        }
	// this shouldn't happen
	if (w == dist.size()) w = eos_sym;

        std::cerr << " " << tdict.Convert(w) << " [p=" << dist[w] << "]";
        t += 1;
        target.push_back(w);
    }
    std::cerr << std::endl;

    return target;
}


}; // namespace cnn
