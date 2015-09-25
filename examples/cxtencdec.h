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
#include "cnn/data-util.h"
#include "dialogue.h"

#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

namespace cnn {

template <class Builder>
class CxtEncDecModel : public DialogueBuilder<Builder>{
public:
    CxtEncDecModel(cnn::Model& model, int vocab_size_src, int layers, int hidden_dim, int hidden_replicates);

public:

    Expression build_graph(const std::vector<int> &source, const std::vector<int>& osent, ComputationGraph &cg);
    Expression build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg){
        return DialogueBuilder<Builder>::build_graph(source, osent, cg);
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg) override;

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg,
            int beam_width, Dict &tdict);

    std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, Dict &tdict);
    
    std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, Dict &tdict);

protected:
    std::vector<Parameters*> p_h0;

};

template <class Builder>
CxtEncDecModel<Builder>::CxtEncDecModel(cnn::Model& model, int vocab_size_src, int layers, int hidden_dim, int hidden_replicates)
    : DialogueBuilder<Builder>(model, vocab_size_src, layers, hidden_dim, hidden_replicates)
{
    /// for context history
    for (auto l = 0; l < hidden_replicates; ++l)
    {
        Parameters *pp = model.add_parameters({ long(hidden_dim) });
        pp->reset_to_zero();
        p_h0.push_back(pp);
    }
}

template <class Builder>
Expression CxtEncDecModel<Builder>::build_graph(const std::vector<int> &source, const std::vector<int>& osent, ComputationGraph &cg)
{
    slen = source.size(); 

    std::vector<Expression> source_embeddings;

    encoder_fwd.new_graph(cg);
    encoder_fwd.start_new_sequence();
    encoder_bwd.new_graph(cg);
    encoder_bwd.start_new_sequence();
    for (int t = 0; t < source.size(); ++t) {
        Expression i_x_t = lookup(cg, p_cs, source[t]);
        encoder_fwd.add_input(i_x_t);
    }
    for (int t = source.size() - 1; t >= 0; --t) {
        Expression i_x_t = lookup(cg, p_cs, source[t]);
        encoder_bwd.add_input(i_x_t);
    }

    cg.incremental_forward();

    /// for contet
    vector<Expression> to;
    /// get the top output
    to.push_back(encoder_fwd.final_h()[layers - 1]);
    to.push_back(encoder_bwd.final_h()[layers - 1]);

    Expression q_m = concatenate(to);

    context.add_input(q_m);
    cg.incremental_forward();

    i_h0.clear();
    for (const auto &p : context.final_s())
        i_h0.push_back(p);

    Expression s_m = concatenate(i_h0);

    // now for the target sentence
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    Expression i_bias_cxt = parameter(cg, p_bias_cxt);

    vector<Expression> oein1, oein2;
    vector<Expression> d_m;
    size_t i = 0;
    for (const auto & p : context.final_s())
    {
        Expression subvec = p + pickrange(i_bias_cxt, i * HIDDEN_DIM, (i + 1)*HIDDEN_DIM);
        for (int i = 0; i < LAYERS; i++)
        {
            oein1.push_back(subvec);
            oein2.push_back(tanh(subvec));
        }
        i++;
    }
    for (int i = 0; i < LAYERS; ++i) d_m.push_back(oein1[i]);
    for (int i = 0; i < LAYERS; ++i) d_m.push_back(oein2[i]);

    decoder.new_graph(cg);
    decoder.start_new_sequence(d_m);

    // decoder
    vector<Expression> errs;

    const unsigned oslen = osent.size() - 1;
    for (unsigned t = 0; t < oslen; ++t) {
        Expression i_x_t = lookup(cg, p_cs, osent[t]);
        Expression i_y_t = decoder.add_input(i_x_t);
        Expression i_r_t = i_bias + i_R * i_y_t;
        Expression i_ydist = log_softmax(i_r_t);
        errs.push_back(pick(i_ydist, osent[t + 1]));
    }
    Expression i_nerr = sum(errs);
    return -i_nerr;
}

template<class Builder>
Expression CxtEncDecModel<Builder>::decoder_step(vector<int> trg_tok, ComputationGraph& cg) 
{
    size_t nutt = trg_tok.size();

    Expression i_x_t;
    vector<Expression> v_x_t;
    for (auto p : trg_tok)
    {
        Expression i_x_x;
        if (p >= 0)
            i_x_x = lookup(cg, p_cs, p);
        else
            i_x_x = input(cg, { (long)hidden_dim }, &zero);
        v_x_t.push_back(i_x_x);
    }
    i_x_t = concatenate_cols(v_x_t);

    Expression i_y_t = decoder.add_input(i_x_t);

    return i_y_t;
}

template <class Builder>
std::vector<int>
CxtEncDecModel<Builder>::decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, cnn::Dict &tdict)
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
    Hypothesis(RNNPointer state, int tgt, float cst, int _t)
        : builder_state(state), target({tgt}), cost(cst), t(_t) {}
    Hypothesis(RNNPointer state, int tgt, float cst, Hypothesis &last)
        : builder_state(state), target(last.target), cost(cst), t(last.t+1) {
        target.push_back(tgt);
    }
    RNNPointer builder_state;
    std::vector<int> target;
    float cost;
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
CxtEncDecModel<Builder>::beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, 
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

    boost::integer_range<int> vocab = boost::irange(0, vocab_size_tgt);
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
CxtEncDecModel<Builder>::sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict &tdict)
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
	double p = rand01();
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
