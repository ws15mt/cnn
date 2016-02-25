#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
//#include "rnnem.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/expr-xtra.h"
#include "cnn/data-util.h"
#include "cnn/dnn.h"
//#include "cnn/decode.h"
//#include "rl.h"
#include "ext/dialogue/dialogue.h"
#include "cnn/approximator.h"

#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

extern int verbose; 

namespace cnn {

#define MEM_SIZE 10
#define REASONING_STEPS 7

template <class Builder, class Decoder>
class AttentionWithIntention : public DialogueBuilder<Builder, Decoder>{
public:
    AttentionWithIntention(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned> & hidden_dim, unsigned hidden_replicates, int additional_input = 0, int mem_slots = 0, cnn::real iscale = 1.0);

    void setAlignDim(cnn::Model& model, unsigned alignd, cnn::real iscale);

    void assign_cxt(ComputationGraph &cg, unsigned int nutt) ;
    void assign_cxt(ComputationGraph &cg, const vector<vector<int>>&) {
        throw("not implemented");
    };

    Expression build_graph(const std::vector<int> &source, const std::vector<int>& target,
        ComputationGraph& cg);

    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg){
        return DialogueBuilder<Builder, Decoder>::build_graph(source, osent, cg);
    };

#ifdef INPUT_UTF8
    std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, Dict<std::wstring> &tdict);
    std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, Dict<std::wstring> &tdict);
#else
    std::vector<int> beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width, Dict &tdict);
    std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg, Dict &tdict);
#endif


    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {

        DialogueBuilder<Builder, Decoder>::start_new_instance(source, cg);

        i_Wa = parameter(cg, p_Wa);
        i_va = parameter(cg, p_va);
        i_Q = parameter(cg, p_Q);

    }

protected:

    Expression i_tgt2cxt;
    Parameters* p_va, *p_Wa;
    Parameters* p_Q;
    Expression i_Wa, i_va, i_Q;
    Expression attention(int trg_tok, ComputationGraph& cg);
    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg) override;
};

template<class Builder, class Decoder>
AttentionWithIntention<Builder, Decoder>::AttentionWithIntention(cnn::Model& model,
    unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers, const vector<unsigned>& hidden_dim, unsigned hidden_replicates, int additional_input, int mem_slots = 0, cnn::real iscale = 1.0)
    : DialogueBuilder<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
{
    /// default uses the same hidden dimenion for alignment dimension
    setAlignDim(model, hidden_dim[ALIGN_LAYER], iscale);
}

template<class Builder, class Decoder>
void AttentionWithIntention<Builder, Decoder>::setAlignDim(cnn::Model& model, unsigned alignd, cnn::real iscale)
{
    unsigned int align_dim = alignd;
    p_Wa = model.add_parameters({ align_dim, layers[DECODER_LAYER] * hidden_dim[DECODER_LAYER] }, iscale);
    p_va = model.add_parameters({ align_dim }, iscale);
    p_Q = model.add_parameters({ hidden_dim[DECODER_LAYER], rep_hidden * hidden_dim[DECODER_LAYER] }, iscale);
}


template<class Builder, class Decoder>
void AttentionWithIntention<Builder, Decoder>::assign_cxt(ComputationGraph &cg, unsigned int nutt){
    i_U = parameter(cg, p_U);
    i_Wa = parameter(cg, p_Wa);
    i_va = parameter(cg, p_va);

    DialogueBuilder<Builder, Decoder>::assign_cxt(cg, nutt);

    if (turnid == 0)
        return;
}

template<class Builder, class Decoder>
Expression AttentionWithIntention<Builder, Decoder>::build_graph(const std::vector<int> &source, const std::vector<int>& osent, ComputationGraph &cg)
{

    start_new_instance(source, cg);

    // decoder
    vector<Expression> errs;

    i_Wa = parameter(cg, p_Wa);
    i_va = parameter(cg, p_va);
    i_U = parameter(cg, p_U);
    i_Q = parameter(cg, p_Q);

    const unsigned oslen = osent.size() - 1;
    for (unsigned t = 0; t < oslen; ++t) {
        Expression i_y_t = attention(osent[t], cg);
        Expression i_r_t = i_bias + i_R * i_y_t;
        Expression i_ydist = log_softmax(i_r_t);
        errs.push_back(pick(i_ydist, osent[t + 1]));
    }

    save_context(cg);

    Expression i_nerr = sum(errs);
    turnid++;
    return -i_nerr;
}

template<class Builder, class Decoder>
Expression AttentionWithIntention<Builder, Decoder>::attention(int trg_tok, ComputationGraph& cg)
{
    vector<int> vi(1, trg_tok);

    return decoder_step(vi, cg); 
}

template<class Builder, class Decoder>
Expression AttentionWithIntention<Builder, Decoder>::decoder_step(vector<int> trg_tok, ComputationGraph& cg)
{
    Expression i_c_t;
    unsigned int nutt = trg_tok.size();
    Expression i_h_tm1 = concatenate(v_decoder.back()->final_h());

    vector<Expression> v_x_t;
    for (auto p : trg_tok)
    {
        Expression i_x_x;
        if (p >= 0)
            i_x_x = lookup(cg, p_cs, p);
        else
            i_x_x = input(cg, { hidden_dim[DECODER_LAYER] }, &zero);
        v_x_t.push_back(i_x_x);
    }
#ifdef UNDERSTAND_AWI_ADD_ATTENTION
    concatenate_cols(v_x_t);

    vector<Expression> alpha;
    vector<Expression> v_obs = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_tm1, src, hidden_dim[ALIGN_LAYER], nutt, alpha, 1.0);

    vector<Expression> v_input;
    for (size_t k = 0; k < nutt; k++)
    {
        v_input.push_back(concatenate(vector<Expression>({ v_x_t[k], v_obs[k] })));
    }
    Expression input = concatenate_cols(v_input);
#else
    Expression input = concatenate_cols(v_x_t);
#endif
    return v_decoder[v_decoder.size() - 1]->add_input(input);
}

template<class Builder, class Decoder>
std::vector<int> AttentionWithIntention<Builder, Decoder>::beam_decode(const std::vector<int> &source, ComputationGraph& cg, int beam_width,
    cnn::Dict &tdict)
{
    const int sos_sym = tdict.Convert("<s>");
    const int eos_sym = tdict.Convert("</s>");

    size_t tgt_len = 2 * source.size();

    start_new_instance(source, cg);

    priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> completed;
    priority_queue<Hypothesis, vector<Hypothesis>, CompareHypothesis> chart;
    chart.push(Hypothesis(builder.state(), sos_sym, 0.0f, 0));

    boost::integer_range<int> vocab = boost::irange(0, vocab_size);
    vector<int> vec_vocab(vocab_size, 0);
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
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

template<class Builder, class Decoder>
std::vector<int> AttentionWithIntention<Builder, Decoder>::sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict &tdict)
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
        auto dist = get_value(ydist, cg);
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

template <class Builder, class Decoder>
class GatedAttention : public AttentionWithIntention<Builder, Decoder>{
public:
    GatedAttention(Model& model,
    unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, int additional_input, int mem_slots = 0, cnn::real iscale = 1.0)
        : AttentionWithIntention(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
    {
        p_att_gate_A = model.add_parameters({ 2 * hidden_dim[DECODER_LAYER], layers[DECODER_LAYER] * hidden_dim[DECODER_LAYER] }, iscale);
        p_att_gate_b = model.add_parameters({ 2 * hidden_dim[DECODER_LAYER] }, iscale);
    }

protected:
    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        unsigned int nutt = source.size();

        AttentionWithIntention<Builder, Decoder>::start_new_instance(source, cg);

        i_att_gate_A = parameter(cg, p_att_gate_A);
        i_att_gate_b = parameter(cg, p_att_gate_b);

        v_att_gate_b = concatenate_cols(vector<Expression>(nutt, i_att_gate_b));
    }

protected:

    Expression i_att_gate_A, i_att_gate_b, v_att_gate_b;
    Parameters* p_att_gate_A, *p_att_gate_b;
    Expression attention_gate(Expression i_h_tm1);
    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg) override;

};

template<class Builder, class Decoder>
Expression GatedAttention<Builder, Decoder>::attention_gate(Expression i_h_tm1)
{
    Expression att = logistic(i_att_gate_A * i_h_tm1 + v_att_gate_b);
    return att;
}

template<class Builder, class Decoder>
Expression GatedAttention<Builder, Decoder>::decoder_step(vector<int> trg_tok, ComputationGraph& cg)
{
    Expression i_c_t;
    unsigned int nutt = trg_tok.size();
    Expression i_h_tm1 = concatenate(decoder.final_h());

    vector<Expression> v_x_t;
    for (auto p : trg_tok)
    {
        Expression i_x_x;
        if (p >= 0)
            i_x_x = lookup(cg, p_cs, p);
        else
            i_x_x = input(cg, { hidden_dim[DECODER_LAYER] }, &zero);
        v_x_t.push_back(i_x_x);
    }

    vector<Expression> alpha;
    vector<Expression> v_obs = attention_to_source(v_src, src_len, i_U, src, i_va, i_Wa, i_h_tm1, hidden_dim[ALIGN_LAYER], nutt, alpha);
    Expression i_att_gate = attention_gate(i_h_tm1);
    Expression i_obs = concatenate_cols(v_obs);
    Expression i_gated_attention = cwise_multiply(i_att_gate , i_obs);
    Expression i_flatted = reshape(i_gated_attention, { nutt * 2 * hidden_dim[DECODER_LAYER] });
    vector<Expression> v_input;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_flatted_element = pickrange(i_flatted, k * 2 * hidden_dim[DECODER_LAYER], (k + 1) * 2 * hidden_dim[DECODER_LAYER]);
        v_input.push_back(concatenate(vector<Expression>({ v_x_t[k], i_flatted_element })));
    }
    Expression input = concatenate_cols(v_input);

    return decoder.add_input(input);
}

template <class Builder, class Decoder>
class AWI : public AttentionWithIntention < Builder, Decoder > {
public:
    AWI(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
        : AttentionWithIntention<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
    {
        p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[ENCODER_LAYER] }, iscale);
    }

    void setAlignDim(cnn::Model& model, unsigned alignd, cnn::real iscale);

    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg)
    {
        unsigned int nutt = source.size();
        start_new_instance(source, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        nutt = osent.size();
        Expression i_bias_mb = concatenate_cols(vector<Expression>(nutt, i_bias));

        int oslen = 0;
        for (auto p : osent)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : osent)
            {
                if (t < p.size()){
                    vobs.push_back(p[t]);
                }
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_bias_mb + i_R * i_y_t;

            Expression x_r_t = reshape(i_r_t, { vocab_size * nutt });
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < osent[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    Expression i_ydist = log_softmax(r_r_t);
                    this_errs[i].push_back( - pick(i_ydist, osent[i][t + 1]));
                    tgt_words++;
                }
                else if (t == osent[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : v_decoder.back()->final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        v_errs.push_back(sum(errs));
        turnid++;
        return errs;
    };

#ifdef INPUT_UTF8
    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict<std::wstring> &tdict)
#else
    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
#endif
    {
#ifdef INPUT_UTF8
        const int sos_sym = tdict.Convert(L"<s>");
        const int eos_sym = tdict.Convert(L"</s>");
#else
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");
#endif

        std::vector<int> target;
        target.push_back(sos_sym);

        int t = 0;

        start_new_single_instance(source, cg);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_bias + i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(v_decoder.back()->final_s());

        save_context(cg);

        turnid++;

        return target;
    }

protected:

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            context.new_graph(cg);
            context.start_new_sequence();
            context.set_data_in_parallel(nutt);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);

            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        v_encoder_bwd.push_back(new Builder(encoder_bwd));

        v_encoder_bwd.back()->new_graph(cg);
        v_encoder_bwd.back()->set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            v_encoder_bwd.back()->start_new_sequence(to_cxt);
        }
        else
            v_encoder_bwd.back()->start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, *v_encoder_bwd.back(), hidden_dim[ENCODER_LAYER]));
        if (verbose)
            display_value(src_fwd, cg, "src_fwd");

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = v_encoder_bwd.back()->final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");

        /// take the top layer from decoder, take its final h
        /// take the top layer from decoder, take its final h
        if (to_cxt.size() > 0)
        {
            Expression i_from_prv_target = concatenate(to_cxt);
            Expression n_q_m = 0.5 * q_m + 0.5 * i_from_prv_target;
            context.add_input(n_q_m);
            if (verbose)
                display_value(n_q_m, cg, "q_m");
            if (verbose)
                display_value(i_from_prv_target, cg, "i_from_prv_target");
            if (verbose)
                display_value(i_tgt2cxt, cg, "i_tgt2cxt");
        }
        else
            context.add_input(q_m);

        i_U = parameter(cg, p_U);
        src = i_U * concatenate_cols(v_src);  // precompute 

        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(i_cxt2dec_w * p);
        v_decoder.push_back(new Decoder(decoder));
        v_decoder.back()->new_graph(cg);
        v_decoder.back()->set_data_in_parallel(nutt);
        v_decoder.back()->start_new_sequence(vcxt);  /// get the intention
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        vector<Expression> v_h = v_decoder.back()->final_h();
        Expression i_h_tm1 = concatenate(v_h);
        if (verbose)
            display_value(i_h_tm1, cg, "i_h_tm1");

        vector<Expression> v_x_t;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = input(cg, { hidden_dim[DECODER_LAYER] }, &zero);
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }
        concatenate_cols(v_x_t);

        vector<Expression> alpha;
        vector<Expression> v_obs = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_tm1, src, hidden_dim[ALIGN_DIM], nutt, alpha, 1.0);
        if (verbose)
            display_value(concatenate_cols(v_obs), cg, "v_obs");
        if (verbose)
            display_value(concatenate_cols(alpha), cg, "alpha");

        vector<Expression> v_input;
        for (size_t k = 0; k < nutt; k++)
        {
            Expression i_obs = concatenate(vector<Expression>({ v_x_t[k], v_obs[k] }));
            v_input.push_back(i_obs);
            if (verbose)
                display_value(i_obs, cg, "i_obs");
        }
        Expression input = concatenate_cols(v_input);

        return v_decoder.back()->add_input(input);
    }

};

template <class Builder, class Decoder>
class AWI_Bilinear : public AWI< Builder , Decoder> {
public:
    AWI_Bilinear(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
    const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
    : AWI<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
    {
        if (hidden_dim[ENCODER_LAYER] != hidden_dim[ALIGN_LAYER])
        {
            cerr << "hidden_dim and align_dim should be the same" << endl;
            throw("hidden_dim and align_dim should be the same ");
        }
        for (size_t k = 0; k < hidden_replicates * layers[DECODER_LAYER]; k++)
        {
            p_tgt2enc_b.push_back(model.add_parameters({ hidden_dim[ENCODER_LAYER] }, iscale));
            p_tgt2enc_w.push_back(model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[DECODER_LAYER] }, iscale));
        }
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg) override;

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            i_tgt2enc_b.clear();
            i_tgt2enc_w.clear();
            context.new_graph(cg);
            context.set_data_in_parallel(nutt);

            if (last_context_exp.size() == 0)
                context.start_new_sequence();
            else
                context.start_new_sequence(last_context_exp);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);

            for (auto p : p_tgt2enc_b)
                i_tgt2enc_b.push_back(parameter(cg, p));
            for (auto p : p_tgt2enc_w)
                i_tgt2enc_w.push_back(parameter(cg, p));

            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        v_encoder_bwd.push_back(new Builder(encoder_bwd));

        v_encoder_bwd.back()->new_graph(cg);
        v_encoder_bwd.back()->set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            if (verbose)
                display_value(concatenate(v_last_decoder_state), cg, "v_last_decoder_state");
            for (size_t k = 0; k < i_tgt2enc_b.size(); k++)
            {
                if (nutt > 1)
                    v_tgt2enc.push_back(concatenate_cols(std::vector<Expression>(nutt, i_tgt2enc_b[k])) + i_tgt2enc_w[k] * to_cxt[k]);
                else 
                    v_tgt2enc.push_back(i_tgt2enc_b[k] + i_tgt2enc_w[k] * to_cxt[k]);
            }
            v_encoder_bwd.back()->start_new_sequence(v_tgt2enc); 
        }
        else
            v_encoder_bwd.back()->start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, *v_encoder_bwd.back(), hidden_dim[ENCODER_LAYER]));
        if (verbose)
            display_value(src_fwd, cg, "src_fwd");

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = v_encoder_bwd.back()->final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");

        /// take the top layer from decoder, take its final h
        if (to_cxt.size() > 0)
        {
            Expression i_from_prv_target = concatenate(to_cxt);
            Expression n_q_m = 0.5 * q_m + 0.5 * i_from_prv_target;
            context.add_input(n_q_m);
            if (verbose)
                display_value(n_q_m, cg, "q_m");
            if (verbose)
                display_value(i_from_prv_target, cg, "i_from_prv_target");
            if (verbose)
                display_value(i_tgt2cxt, cg, "i_tgt2cxt");
        }
        else
            context.add_input(q_m);

        i_U = parameter(cg, p_U);
        src = i_U * concatenate_cols(v_src);  // precompute 

        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(i_cxt2dec_w * p);
        v_decoder.push_back(new Decoder(decoder));
        v_decoder.back()->new_graph(cg);
        v_decoder.back()->set_data_in_parallel(nutt);
        v_decoder.back()->start_new_sequence(vcxt);  /// get the intention
    }

    public:
        vector<Expression> i_tgt2enc_b, i_tgt2enc_w;
    public:    
        vector<Parameters*> p_tgt2enc_b;
        vector<Parameters*> p_tgt2enc_w;

};

template<class Builder, class Decoder>
Expression AWI_Bilinear<Builder, Decoder>::decoder_step(vector<int> trg_tok, ComputationGraph& cg)
{
    Expression i_c_t;
    unsigned int nutt = trg_tok.size();
    vector<Expression> v_h = v_decoder.back()->final_h();
    Expression i_h_tm1 = concatenate(v_h);
    if (verbose)
        display_value(i_h_tm1, cg, "i_h_tm1");

    vector<Expression> v_x_t;
    for (auto p : trg_tok)
    {
        Expression i_x_x;
        if (p >= 0)
            i_x_x = lookup(cg, p_cs, p);
        else
            i_x_x = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);
        if (verbose)
            display_value(i_x_x, cg, "i_x_x");
        v_x_t.push_back(i_x_x);
    }
    concatenate_cols(v_x_t);

    vector<Expression> alpha;
    vector<Expression> v_obs = attention_to_source_bilinear(v_src, src_len, i_va, i_Wa, i_h_tm1, hidden_dim[ALIGN_LAYER], nutt, alpha, 2.0);
    if (verbose)
        display_value(concatenate_cols(alpha), cg, "alpha");

    vector<Expression> v_input;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_obs = concatenate(vector<Expression>({ v_x_t[k], v_obs[k] }));
        v_input.push_back(i_obs);
        if (verbose)
            display_value(i_obs, cg, "i_obs");
    }
    Expression input = concatenate_cols(v_input);;
    if (verbose)
        display_value(concatenate_cols(v_obs), cg, "v_obs");

    return v_decoder.back()->add_input(input);
}

template <class Builder, class Decoder>
class AWI_Bilinear_Simpler : public AWI_Bilinear < Builder, Decoder> {
public:
    AWI_Bilinear_Simpler(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
    const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
    : AWI_Bilinear<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
    {
    }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            i_tgt2enc_b.clear();
            i_tgt2enc_w.clear();
            context.new_graph(cg);
            context.set_data_in_parallel(nutt);

            if (last_context_exp.size() == 0)
                context.start_new_sequence();
            else
                context.start_new_sequence(last_context_exp);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);
            for (auto p : p_tgt2enc_b)
                i_tgt2enc_b.push_back(parameter(cg, p));
            for (auto p : p_tgt2enc_w)
                i_tgt2enc_w.push_back(parameter(cg, p));

            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        v_encoder_bwd.push_back(new Builder(encoder_bwd));

        v_encoder_bwd.back()->new_graph(cg);
        v_encoder_bwd.back()->set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            if (verbose)
                display_value(concatenate(v_last_decoder_state), cg, "v_last_decoder_state");
            for (size_t k = 0; k < i_tgt2enc_b.size(); k++)
            {
                if (nutt > 1)
                    v_tgt2enc.push_back(concatenate_cols(std::vector<Expression>(nutt, i_tgt2enc_b[k])) + i_tgt2enc_w[k] * to_cxt[k]);
                else
                    v_tgt2enc.push_back(i_tgt2enc_b[k] + i_tgt2enc_w[k] * to_cxt[k]);
            }
            v_encoder_bwd.back()->start_new_sequence(v_tgt2enc);
        }
        else
            v_encoder_bwd.back()->start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, *v_encoder_bwd.back(), hidden_dim[ENCODER_LAYER]));
        if (verbose)
            display_value(src_fwd, cg, "src_fwd");

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = v_encoder_bwd.back()->final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");

        context.add_input(q_m);

        i_U = parameter(cg, p_U);
        src = i_U * concatenate_cols(v_src);  // precompute 

        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(i_cxt2dec_w * p);
        v_decoder.push_back(new Decoder(decoder));
        v_decoder.back()->new_graph(cg);
        v_decoder.back()->set_data_in_parallel(nutt);
        v_decoder.back()->start_new_sequence(vcxt);  /// get the intention
    }
};

/**
inspired from the paper
"Effective approaches to attention-based neural machine translation" by M. Luong, H. Pham, and C. Manning, EMNLP 2015
http://nlp.stanford.edu/pubs/emnlp15_attn.pdf
use local attention, generic, and input feeding

Only the top-layer LSTM is used. Input is reversed. Attention layer takes the top layer from encoder and decoder. Decoder initialized from
context layer. 
Local attention: 
a_t(s) = align(h_t, \bar{h}_s) exp( - (s-p_t)^2/(2 sigma^2)
p_t = S * sigmoid(v_p^T tanh(W_p h_t))

Attention:
a_t(s) = align(h_t, \bar{h}_s) = 
exp(score(h_t, \bar{h}_s))/(normalization)
score(h_t, \bar{h}_s) = h_t^T W_a \bar{h}_s
c_t = \sum_s a_t(s) \bar{h}_s

Decoder:
\tilde{h}_t = tanh(W_c[c_t; h_t])

Input feeding
h_t = f(h_{t-1}, y_{t-1}, \tilde{h}_{t-1})
*/
template <class Builder, class Decoder>
class AWI_LocalGeneralInputFeeding: public AWI_Bilinear_Simpler< Builder, Decoder> {
private:
    /// for location prediction for local attention
    Parameters* p_va_local, *p_Wa_local, *p_ba_local;
    Expression i_va_local, i_Wa_local, i_ba_local;

    /// for the expoential scaling of the softmax
    Parameters* p_scale;
    Expression i_scale; 

    /// a single layer MLP
    DNNBuilder attention_layer;
    vector<Expression> attention_output_for_this_turn; /// [number of turn]

    Expression i_zero;

public:
    AWI_LocalGeneralInputFeeding(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
    const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
    : AWI_Bilinear_Simpler<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale),
    attention_layer(1, vector<unsigned>{hidden_dim[ENCODER_LAYER] + hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER]}, &model, iscale, "attention_layer")
    {
        p_Wa_local = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[DECODER_LAYER] }, iscale);
        p_ba_local = model.add_parameters({ hidden_dim[ALIGN_LAYER] }, iscale);
        p_va_local = model.add_parameters({ 1, hidden_dim[ALIGN_LAYER] }, iscale);

        p_Wa = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[DECODER_LAYER] }, iscale);

        p_scale = model.add_parameters({ 1 }, 0.0);
    }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            attention_output_for_this_turn.clear();

            i_tgt2enc_b.clear();
            i_tgt2enc_w.clear();
            context.new_graph(cg);

            if (last_context_exp.size() == 0)
                context.start_new_sequence();
            else
                context.start_new_sequence(last_context_exp);
            context.set_data_in_parallel(nutt);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            i_Wa_local = parameter(cg, p_Wa_local);
            i_va_local = parameter(cg, p_va_local);
            i_ba_local = parameter(cg, p_ba_local);

            i_scale = exp(parameter(cg, p_scale));

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);
            for (auto p : p_tgt2enc_b)
                i_tgt2enc_b.push_back(parameter(cg, p));
            for (auto p : p_tgt2enc_w)
                i_tgt2enc_w.push_back(parameter(cg, p));

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            /// encoder starts with the last decoder's state
            v_tgt2enc.resize(i_tgt2enc_b.size());

            for (int k = 0; k < i_tgt2enc_b.size(); k++)
            {
                if (nutt > 1)
                    v_tgt2enc[k] = concatenate_cols(std::vector<Expression>(nutt, i_tgt2enc_b[k])) + i_tgt2enc_w[k] * to_cxt[k];
                else
                    v_tgt2enc[k] = i_tgt2enc_b[k] + i_tgt2enc_w[k] * to_cxt[k];
            }
            encoder_bwd.start_new_sequence(v_tgt2enc);
        }
        else
            encoder_bwd.start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the backward direction encoding of the source
        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, encoder_bwd, hidden_dim[ENCODER_LAYER]));

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = encoder_bwd.final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");
        /// update intention
        context.add_input(q_m);

        /// decoder start with a context from intention 
        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(i_cxt2dec_w * p);
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        decoder.start_new_sequence(vcxt);  /// get the intention
    }

    /// return errors[nutt]
    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg)
    {
        unsigned int nutt = source.size();
        start_new_instance(source, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        nutt = osent.size();

        int oslen = 0;
        for (auto p : osent)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : osent)
            {
                if (t < p.size()){
                    vobs.push_back(p[t]);
                }
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_R * i_y_t;

            Expression x_r_t = reshape(i_r_t, { vocab_size * nutt });
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < osent[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    Expression i_ydist = log_softmax(r_r_t);
                    this_errs[i].push_back(-pick(i_ydist, osent[i][t + 1]));
                    tgt_words++;
                }
                else if (t == osent[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero; 
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else 
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        vector<Expression> alpha;
        vector<Expression> position = local_attention_to(cg, src_len, i_Wa_local, i_ba_local, i_va_local, i_h_t, nutt);
        if (verbose)
        {
            for (auto &p : position){
                display_value(p, cg, "predicted local attention position ");
            }
        }

        vector<Expression> v_context_to_source = attention_using_bilinear_with_local_attention(v_src, src_len, i_Wa, i_h_t, hidden_dim[ENCODER_LAYER], nutt, alpha, i_scale, position);
        if (verbose)
        {
            size_t k = 0;
            for (auto &p : alpha){
                display_value(p, cg, "attention_to_source_weight_" + boost::lexical_cast<string>(k++));
            }
        }

        /// compute attention
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenate_cols(v_context_to_source) });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        return i_h_attention_t;
    }
};

/**
inspired from the paper
"Effective approaches to attention-based neural machine translation" by M. Luong, H. Pham, and C. Manning, EMNLP 2015
http://nlp.stanford.edu/pubs/emnlp15_attn.pdf
use generic, and input feeding
dont use local attention, as it seems not producing reasonable prediction of source side positions

Only the top-layer LSTM is used. Input is reversed. Attention layer takes the top layer from encoder and decoder. Decoder initialized from
context layer.

Attention:
a_t(s) = align(h_t, \bar{h}_s) =
exp(score(h_t, \bar{h}_s))/(normalization)
score(h_t, \bar{h}_s) = h_t^T W_a \bar{h}_s
c_t = \sum_s a_t(s) \bar{h}_s

Decoder:
\tilde{h}_t = tanh(W_c[c_t; h_t])

Input feeding
h_t = f(h_{t-1}, y_{t-1}, \tilde{h}_{t-1})
*/
template <class Builder, class Decoder>
class AWI_GeneralInputFeeding : public AWI_Bilinear_Simpler< Builder, Decoder> {
protected:
    /// for the expoential scaling of the softmax
    Parameters* p_scale;
    Expression i_scale;

    /// a single layer MLP
    DNNBuilder attention_layer;
    vector<Expression> attention_output_for_this_turn; /// [number of turn]

    Expression i_zero;

public:
    AWI_GeneralInputFeeding(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
        : AWI_Bilinear_Simpler<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale),
        attention_layer(1, vector<unsigned>{hidden_dim[ENCODER_LAYER] + hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER]}, &model, iscale, "attention_layer")
    {
            p_Wa = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[DECODER_LAYER] }, iscale);
            p_scale = model.add_parameters({ 1 }, 0.0);
        }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            attention_output_for_this_turn.clear();

            i_tgt2enc_b.clear();
            i_tgt2enc_w.clear();
            context.new_graph(cg);

            if (last_context_exp.size() == 0)
                context.start_new_sequence();
            else
                context.start_new_sequence(last_context_exp);
            context.set_data_in_parallel(nutt);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            i_scale = exp(parameter(cg, p_scale));

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);
            for (auto p : p_tgt2enc_b)
                i_tgt2enc_b.push_back(parameter(cg, p));
            for (auto p : p_tgt2enc_w)
                i_tgt2enc_w.push_back(parameter(cg, p));

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            /// encoder starts with the last decoder's state
            for (size_t k = 0; k < i_tgt2enc_b.size(); k++)
            {
                if (nutt > 1)
                    v_tgt2enc.push_back(concatenate_cols(std::vector<Expression>(nutt, i_tgt2enc_b[k])) + i_tgt2enc_w[k] * to_cxt[k]);
                else
                    v_tgt2enc.push_back(i_tgt2enc_b[k] + i_tgt2enc_w[k] * to_cxt[k]);
            }
            encoder_bwd.start_new_sequence(v_tgt2enc);
        }
        else
            encoder_bwd.start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the backward direction encoding of the source
        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, encoder_bwd, hidden_dim[ENCODER_LAYER]));

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = encoder_bwd.final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");
        /// update intention
        context.add_input(q_m);

        /// decoder start with a context from intention 
        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(i_cxt2dec_w * p);
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        decoder.start_new_sequence(vcxt);  /// get the intention
    }

    /// return errors[nutt]
    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg)
    {
        unsigned int nutt = source.size();
        start_new_instance(source, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        nutt = osent.size();

        int oslen = 0;
        for (auto p : osent)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : osent)
            {
                if (t < p.size()){
                    vobs.push_back(p[t]);
                }
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_R * i_y_t;
            Expression i_ydist = log_softmax(i_r_t);
            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (size_t i = 0; i < nutt; i++)
            {
                size_t offset = i * vocab_size; 
                if (t < osent[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, osent[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == osent[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_using_bilinear(v_src, src_len, i_Wa, i_h_t, hidden_dim[ENCODER_LAYER], nutt, alpha, i_scale);
        if (verbose)
        {
            size_t k = 0;
            for (auto &p : alpha){
                display_value(p, cg, "attention_to_source_weight_" + boost::lexical_cast<string>(k++));
            }
        }

        /// compute attention
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenate_cols(v_context_to_source) });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        return i_h_attention_t;
    }
};

/**
inspired from the paper
"Effective approaches to attention-based neural machine translation" by M. Luong, H. Pham, and C. Manning, EMNLP 2015
http://nlp.stanford.edu/pubs/emnlp15_attn.pdf
use generic, and input feeding
dont use local attention, as it seems not producing reasonable prediction of source side positions

Only the top-layer LSTM is used. Input is reversed. Attention layer takes the top layer from encoder and decoder. Decoder initialized from
context layer.

Attention:
a_t(s) = align(h_t, \bar{h}_s) =
exp(score(h_t, \bar{h}_s))/(normalization)
score(h_t, \bar{h}_s) = h_t^T W_a \bar{h}_s
c_t = \sum_s a_t(s) \bar{h}_s

Decoder:
\tilde{h}_t = tanh(W_c[c_t; h_t])

Input feeding
h_t = f(h_{t-1}, y_{t-1}, \tilde{h}_{t-1})
*/
template <class Builder, class Decoder>
class AWI_GeneralInputFeedingWDropout : public AWI_GeneralInputFeeding< Builder, Decoder> {
protected:
    /// dropout rate
    cnn::real dropout_rate; 
public:
    AWI_GeneralInputFeedingWDropout(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
        : AWI_GeneralInputFeeding<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
    {
            dropout_rate = 0.5;
        }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            attention_output_for_this_turn.clear();

            i_tgt2enc_b.clear();
            i_tgt2enc_w.clear();
            context.new_graph(cg);

            if (last_context_exp.size() == 0)
                context.start_new_sequence();
            else
                context.start_new_sequence(last_context_exp);
            context.set_data_in_parallel(nutt);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            i_scale = exp(parameter(cg, p_scale));

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);
            for (auto p : p_tgt2enc_b)
                i_tgt2enc_b.push_back(parameter(cg, p));
            for (auto p : p_tgt2enc_w)
                i_tgt2enc_w.push_back(parameter(cg, p));

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            /// encoder starts with the last decoder's state
            for (size_t k = 0; k < i_tgt2enc_b.size(); k++)
            {
                if (nutt > 1)
                    v_tgt2enc.push_back(concatenate_cols(std::vector<Expression>(nutt, i_tgt2enc_b[k])) + i_tgt2enc_w[k] * to_cxt[k]);
                else
                    v_tgt2enc.push_back(i_tgt2enc_b[k] + i_tgt2enc_w[k] * to_cxt[k]);
            }
            encoder_bwd.start_new_sequence(v_tgt2enc);
        }
        else
            encoder_bwd.start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the backward direction encoding of the source
        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, encoder_bwd, hidden_dim[ENCODER_LAYER]));

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = encoder_bwd.final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");
        /// update intention
        context.add_input(q_m);

        /// decoder start with a context from intention 
        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(dropout(i_cxt2dec_w * p, dropout_rate)); /// drop out from intention
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        decoder.start_new_sequence(vcxt);  /// get the intention
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, dropout(attention_output_for_this_turn.back(), dropout_rate) });
        }

        i_h_t = decoder.add_input(i_input);

        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_using_bilinear(v_src, src_len, i_Wa, dropout(i_h_t, dropout_rate), hidden_dim[ENCODER_LAYER], nutt, alpha, i_scale);
        if (verbose)
        {
            size_t k = 0;
            for (auto &p : alpha){
                display_value(p, cg, "attention_to_source_weight_" + boost::lexical_cast<string>(k++));
            }
        }

        /// compute attention
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenate_cols(v_context_to_source) });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        return i_h_attention_t;
    }
};

/**
inspired from the paper

Only the top-layer LSTM is used. Input is reversed. Attention layer takes the top layer from encoder and decoder. Decoder initialized from
context layer.

Attention layer is a 2-layer feed-forward network

Decoder:
\tilde{h}_t = tanh(W_c[c_t; h_t])

Input feeding
h_t = f(h_{t-1}, y_{t-1}, \tilde{h}_{t-1})
*/
template <class Builder, class Decoder>
class AWI_InputFeedingWithNNAttention: public AWI_GeneralInputFeedingWDropout< Builder, Decoder> {
protected:
    // a single layer MLP
    ReluDNNBuilder true_attention_layer;
    DNNBuilder response_layer;

public:
    AWI_InputFeedingWithNNAttention(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
        : AWI_GeneralInputFeedingWDropout<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale),
        true_attention_layer(2, vector<unsigned>{hidden_dim[ENCODER_LAYER] + hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER], 1}, &model, iscale, "true attention layer"),
        response_layer(1, vector<unsigned>{hidden_dim[ENCODER_LAYER] + hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER]}, &model, iscale, "response layer")
    {
        }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            attention_output_for_this_turn.clear();

            i_tgt2enc_b.clear();
            i_tgt2enc_w.clear();
            context.new_graph(cg);

            if (last_context_exp.size() == 0)
                context.start_new_sequence();
            else
                context.start_new_sequence(last_context_exp);
            context.set_data_in_parallel(nutt);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            i_scale = exp(parameter(cg, p_scale));

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);
            for (auto p : p_tgt2enc_b)
                i_tgt2enc_b.push_back(parameter(cg, p));
            for (auto p : p_tgt2enc_w)
                i_tgt2enc_w.push_back(parameter(cg, p));

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            true_attention_layer.new_graph(cg);
            true_attention_layer.set_data_in_parallel(nutt);
            response_layer.new_graph(cg);
            response_layer.set_data_in_parallel(nutt);
            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            /// encoder starts with the last decoder's state
            for (size_t k = 0; k < i_tgt2enc_b.size(); k++)
            {
                if (nutt > 1)
                    v_tgt2enc.push_back(concatenate_cols(std::vector<Expression>(nutt, i_tgt2enc_b[k])) + i_tgt2enc_w[k] * to_cxt[k]);
                else
                    v_tgt2enc.push_back(i_tgt2enc_b[k] + i_tgt2enc_w[k] * to_cxt[k]);
            }
            encoder_bwd.start_new_sequence(v_tgt2enc);
        }
        else
            encoder_bwd.start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the backward direction encoding of the source
        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, encoder_bwd, hidden_dim[ENCODER_LAYER]));

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = encoder_bwd.final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");
        /// update intention
        context.add_input(q_m);

        /// decoder start with a context from intention 
        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(i_cxt2dec_w * p);
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        decoder.start_new_sequence(vcxt);  /// get the intention
    }

    /** use a nn to extraction attention
    */
    vector<Expression> attention_to_source_using_nn(vector<Expression> & v_src, const vector<unsigned>& v_slen, 
        Expression i_h_tm1, unsigned int a_dim, unsigned int nutt, vector<Expression>& v_wgt, Expression fscale)
    {
        Expression i_c_t;
        Expression i_e_t;
        unsigned int slen = 0;
        vector<Expression> i_wah_rep;

        for (auto p : v_slen)
            slen += p;

        Expression i_wah = i_h_tm1;  /// [d nutt]
        Expression i_wah_reshaped = reshape(i_wah, { nutt * a_dim });
        for (size_t k = 0; k < nutt; k++)
        {
            Expression i_wah_each = pickrange(i_wah_reshaped, k * a_dim, (k + 1)*a_dim);  /// [d]
            /// need to do subsampling
            Expression i_tgt = concatenate_cols(std::vector<Expression>(v_slen[k], i_wah_each));
            vector<Expression> v_tgt_src;
            v_tgt_src.push_back(i_tgt);
            v_tgt_src.push_back(v_src[k]);
            Expression i_tgt_src = concatenate(v_tgt_src); /// combine both source and target
            i_wah_rep.push_back(i_tgt_src);  /// [d v_slen[k]]
        }
        Expression i_wah_m = concatenate_cols(i_wah_rep);  // [(d_s + d_t) \sum_k v_slen[k] ]

        true_attention_layer.set_data_in_parallel(slen);
        i_e_t = true_attention_layer.add_input(i_wah_m); 
        i_e_t = reshape(i_e_t, { slen, 1 });
    
        Expression i_alpha_t;

        vector<Expression> v_input;
        int istt = 0;
        for (size_t k = 0; k < nutt; k++)
        {
            Expression i_input;
            int istp = istt + v_slen[k];

            vector<Expression> vscale(v_slen[k], fscale);

            Expression wgt = softmax(cwise_multiply(concatenate(vscale),  pickrange(i_e_t, istt, istp)));
            v_wgt.push_back(wgt);

            i_input = v_src[k] * wgt;  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
            v_input.push_back(i_input);

            istt = istp;
        }

        return v_input;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source_using_nn(v_src, src_len, i_h_t, hidden_dim[DECODER_LAYER], nutt, alpha, i_scale);
            
        if (verbose)
        {
            size_t k = 0;
            for (auto &p : alpha){
                display_value(p, cg, "attention_to_source_weight_" + boost::lexical_cast<string>(k++));
            }
        }

        /// compute response
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenate_cols(v_context_to_source) });
        i_h_attention_t = response_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        return i_h_attention_t;
    }
};

/**
No attention for comparison
use hirearchical process, with an intention network, encoder network and decoder network. however, no attention to encoder output is used when computing decoder network. this is just for comparison to AWI_Bilinear_Simpler, as this simplified model is not the right model to pursue.
*/
template<class Builder, class Decoder>
class HirearchicalEncDec : public AWI_Bilinear_Simpler< Builder, Decoder > {
public:
    HirearchicalEncDec(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
        : AWI_Bilinear_Simpler<Builder, Decoder>(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
    {
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        vector<Expression> v_h = v_decoder.back()->final_h();
        Expression i_h_tm1 = concatenate(v_h);
        if (verbose)
            display_value(i_h_tm1, cg, "i_h_tm1");

        vector<Expression> v_x_t;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = input(cg, { hidden_dim[DECODER_LAYER] }, &zero);
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }
        Expression i_obs = concatenate_cols(v_x_t);

        return v_decoder.back()->add_input(i_obs);
    }
};

/// + autoencoding 
/// the intention vector needs to generate the original source side sentence
template<class Builder, class Decoder>
class AWI_Bilinear_Simpler_AE : public AWI_Bilinear_Simpler< Builder, Decoder > {
public:
    AWI_Bilinear_Simpler_AE(Model& model,
        unsigned vocab_size_src, const vector<unsigned int>& layers,
    const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
    : AWI_Bilinear_Simpler<Builder, Decoder>(model, vocab_size_src, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
    {
    }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) override
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            i_tgt2enc_b.clear();
            i_tgt2enc_w.clear();
            context.new_graph(cg);
            context.set_data_in_parallel(nutt);

            if (last_context_exp.size() == 0)
                context.start_new_sequence();
            else
                context.start_new_sequence(last_context_exp);

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter

            for (auto p : p_tgt2enc_b)
                i_tgt2enc_b.push_back(parameter(cg, p));
            for (auto p : p_tgt2enc_w)
                i_tgt2enc_w.push_back(parameter(cg, p));

            if (verbose)
                display_value(concatenate(i_h0), cg, "i_h0");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        v_encoder_bwd.push_back(new Builder(encoder_bwd));
        v_encoder_bwd.back()->new_graph(cg);
        v_encoder_bwd.back()->set_data_in_parallel(nutt);
        if (to_cxt.size() > 0)
        {
            if (verbose)
                display_value(concatenate(v_last_decoder_state), cg, "v_last_decoder_state");
            for (size_t k = 0; k < i_tgt2enc_b.size(); k++)
            {
                if (nutt > 1)
                    v_tgt2enc.push_back(concatenate_cols(std::vector<Expression>(nutt, i_tgt2enc_b[k])) + i_tgt2enc_w[k] * to_cxt[k]);
                else
                    v_tgt2enc.push_back(i_tgt2enc_b[k] + i_tgt2enc_w[k] * to_cxt[k]);
            }
            v_encoder_bwd.back()->start_new_sequence(v_tgt2enc);
        }
        else
            v_encoder_bwd.back()->start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, *v_encoder_bwd.back(), hidden_dim[ENCODER_LAYER]));
        if (verbose)
            display_value(src_fwd, cg, "src_fwd");

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);
        if (verbose)
            display_value(concatenate_cols(v_src), cg, "v_src");

        /// have input to context RNN
        vector<Expression> to = v_encoder_bwd.back()->final_s();

        Expression q_m = concatenate(to);
        if (verbose)
            display_value(q_m, cg, "q_m");

        context.add_input(q_m);

        i_U = parameter(cg, p_U);
        src = i_U * concatenate_cols(v_src);  // precompute 

        vector<Expression> vcxt;
        for (auto p : context.final_s())
            vcxt.push_back(i_cxt2dec_b + i_cxt2dec_w * p);
        v_decoder.push_back(new Decoder(decoder));
        v_decoder.back()->new_graph(cg);
        v_decoder.back()->set_data_in_parallel(nutt);
        v_decoder.back()->start_new_sequence(vcxt);  /// get the intention

        /// use the same encoder model to generate source side sentence, but 
        /// is initialised from the intention vector
        v_encoder_fwd.push_back(new Builder(encoder_bwd));
        v_encoder_fwd.back()->new_graph(cg);
        v_encoder_fwd.back()->set_data_in_parallel(nutt);
        v_encoder_fwd.back()->start_new_sequence(v_cxt);
    }

    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg)
    {
        unsigned int nutt = source.size();
        start_new_instance(source, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, *v_encoder_fwd.back(), hidden_dim[ENCODER_LAYER]));
        if (verbose)
            display_value(src_fwd, cg, "src_fwd");

        nutt = osent.size();

        int oslen = 0;
        for (auto p : osent)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        Expression i_bias_mb = concatenate_cols(vector<Expression>(nutt, i_bias));

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : osent)
            {
                if (t < p.size()){
                    vobs.push_back(p[t]);
                }
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_bias_mb + i_R * i_y_t;

            Expression x_r_t = reshape(i_r_t, { vocab_size * nutt });
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < osent[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    Expression i_ydist = log_softmax(r_r_t);
                    this_errs[i].push_back(-pick(i_ydist, osent[i][t + 1]));
                    tgt_words++;
                }
                else if (t == osent[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : v_decoder.back()->final_s())
                    {
                        Expression i_tt = reshape(p, { nutt * hidden_dim[DECODER_LAYER] });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };
};

/*
#define RL_ERROR_WEIGHT 1e-3
/// this makes the RL error 1/10 of the value as xent
/// exp(4) = 57 (PPL)
/// 400/4 = 1e-2
template <class Builder>
class AWI_ReinforcementLearning : public AWI_Bilinear< Builder > 
{
    RLCritic<Builder> mRLcritic; /// the critic for each turn

    Expression i_suptrain_err; /// supervised training error
    Expression i_rl_err; /// reinforcement learning error

    explicit AWI_ReinforcementLearning(Model& model,
        unsigned vocab_size_src, const vector<size_t>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
        : AWI_Bilinear<Builder>(model, vocab_size_src, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale),
        mRLcritic(model, layers, hidden_dim[DECODER_LAYER], hidden_replicates, iscale)
    {
    }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) 
    {
        AWI_Bilinear::start_new_instance(source, cg);
        mRLcritic.add_intention(concatenate(context.final_s()));
    }

    Expression build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg)
    {
        i_suptrain_err = AWI_Bilinear::build_graph(source, osent, cg);
        mRLcritic.add_decoder_state(concatenate(to_cxt));
        return i_suptrain_err;
    };

    /// for context
    void reset()
    {
        DialogueBuilder::reset(); 

        mRLcritic.reset(); 
    }

    Expression rl_build_graph(ComputationGraph& cg)
    {
        cg.forward();

        vector<cnn::real> i_cxt;  /// to-do : need to assign to the cross entropy at each time
        for (auto p : v_errs)
        {
            vector<cnn::real> rv = get_value(p, cg);
            if (rv.size() != 1)
            {
                cerr << "should be one dimension" << endl;
            }
            i_cxt.push_back(-1.0 *  rv[0]);  /// prop to log prob, notice that the term in rv is the neg log prob, so need to times -1.0 to convert it back to log prob
        }
        i_rl_err = mRLcritic.build_graph(i_cxt, cg);
        return RL_ERROR_WEIGHT * i_rl_err + i_suptrain_err;
    }
};

template <class Builder>
class AttentionToExtMem : public GatedAttention< Builder > {
protected: 

    std::vector<Parameters*> p_h0;
    vector<Expression> i_h0;
    size_t m_hidden_replicates;
    RNNEMBuilder m_mem_network;
    unordered_map<int, vector<vector<cnn::real>>> last_intent_cxt_s;  /// memory of context history for intent
    
public:

    explicit AttentionToExtMem(Model& model,
        unsigned vocab_size_src, const vector<size_t>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, int additional_input = 2, int mem_slots = MEM_SIZE, cnn::real iscale = 1.0)
        : GatedAttention(model, vocab_size_src, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale),
        m_mem_network(layers), hidden_dim[ENCODER_LAYER], hidden_dim[ENCODER_LAYER], hidden_dim[ALIGN_LAYER], { hidden_dim[ALIGN_LAYER], mem_slots) }, { hidden_dim[ALIGN_LAYER], mem_slots) }, &model, iscale), m_hidden_replicates(hidden_replicates)
    {
        unsigned align_dim = hidden_dim[ALIGN_LAYER];
        size_t k = 0;
        vector<Parameters*> v_c, v_h, v_M, v_K;
        size_t l = 0;
        while (l < layers)
        {
            k = 0;
            while (k < hidden_replicates - 2)
            {
                Parameters *pp = model.add_parameters({ hidden_dim[ENCODER_LAYER] }, iscale);
                if (k==0)
                    v_c.push_back(pp);
                if (k == 1)
                    v_h.push_back(pp);
                k++;
            }

            if (k > hidden_replicates - 1)
                break;
            Parameters *pp = model.add_parameters({ align_dim, mem_slots }, iscale);  /// key 
            v_M.push_back(pp);
            k++;

            if (k > hidden_replicates - 1)
                break;
            Parameters *pv = model.add_parameters({ align_dim, mem_slots }, iscale);  /// value
            v_K.push_back(pv);
            k++;

            l++;
        }

        for (auto p : v_c) p_h0.push_back(p);
        for (auto p : v_h) p_h0.push_back(p);
        for (auto p : v_M) p_h0.push_back(p);
        for (auto p : v_K) p_h0.push_back(p);
    }


public:
    void assign_cxt(ComputationGraph &cg, unsigned int nutt) override
    {

        GatedAttention<Builder>::assign_cxt(cg, nutt);

        if (turnid == 0)
        {
            m_mem_network.new_graph(cg);
            i_h0.clear();
            size_t k = 0, j = 0;
            vector<Expression> v_c, v_h, v_M, v_K;
            for (size_t k = 0; k < m_hidden_replicates - 2; k++)
            {
                if (k == 0)
                {
                    for (size_t l = 0; l < layers; l++)
                    {
                        v_c.push_back(parameter(cg, p_h0[j++]));
                    }
                }
                if (k == 1)
                {
                    for (size_t l = 0; l < layers; l++)
                    {
                        v_h.push_back(parameter(cg, p_h0[j++]));
                    }
                }
            }
            for (size_t l = 0; l < layers; l++)
            {
                v_M.push_back(parameter(cg, p_h0[j++]));
            }
            for (size_t l = 0; l < layers; l++)
            {
                v_K.push_back(parameter(cg, p_h0[j++]));
            }
            for (auto p : v_c) i_h0.push_back(concatenate_cols(vector<Expression>(nutt, p)));
            for (auto p : v_h) i_h0.push_back(concatenate_cols(vector<Expression>(nutt, p)));
            for (auto p : v_M) i_h0.push_back(p);
            for (auto p : v_K) i_h0.push_back(p);

            m_mem_network.start_new_sequence(i_h0);
            return;
        }

        vector<Expression> ve;
        size_t k = 0;
        for (const auto &p : last_intent_cxt_s[turnid - 1])
        {
            Expression iv;
            if (k < (m_hidden_replicates - 2) * layers)
            {
                if (nutt > 1)
                    iv = input(cg, { p.size() / nutt, nutt }, &p);
                else
                    iv = input(cg, { p.size() }, &p);
            }
            else
            {
                if (k < (m_hidden_replicates - 1) * layers)
                    iv = input(cg, m_mem_network.m_mem_dim, &p);
                else 
                    iv = input(cg, m_mem_network.m_key_dim, &p);
            }
            ve.push_back(iv);
            k++;
        }

        m_mem_network.final_s() = ve; 
    }

    void assign_cxt(ComputationGraph &cg, unsigned int nutt,
        vector<vector<cnn::real>>& v_last_s, vector<vector<cnn::real>>& v_decoder_s) override
    {

        GatedAttention<Builder>::assign_cxt(cg, nutt);

        if (turnid == 0)
        {
            m_mem_network.new_graph(cg);
            i_h0.clear();
            size_t k = 0, j = 0;
            vector<Expression> v_c, v_h, v_M, v_K;
            for (size_t k = 0; k < m_hidden_replicates - 2; k++)
            {
                if (k == 0)
                {
                    for (size_t l = 0; l < layers; l++)
                    {
                        v_c.push_back(parameter(cg, p_h0[j++]));
                    }
                }
                if (k == 1)
                {
                    for (size_t l = 0; l < layers; l++)
                    {
                        v_h.push_back(parameter(cg, p_h0[j++]));
                    }
                }
            }
            for (size_t l = 0; l < layers; l++)
            {
                v_M.push_back(parameter(cg, p_h0[j++]));
            }
            for (size_t l = 0; l < layers; l++)
            {
                v_K.push_back(parameter(cg, p_h0[j++]));
            }
            for (auto p : v_c) i_h0.push_back(concatenate_cols(vector<Expression>(nutt, p)));
            for (auto p : v_h) i_h0.push_back(concatenate_cols(vector<Expression>(nutt, p)));
            for (auto p : v_M) i_h0.push_back(p);
            for (auto p : v_K) i_h0.push_back(p);

            m_mem_network.start_new_sequence(i_h0);
            return;
        }

        vector<Expression> ve;
        size_t k = 0;
        for (const auto &p : last_intent_cxt_s[turnid - 1])
        {
            Expression iv;
            if (k < (m_hidden_replicates - 2) * layers)
            {
                if (nutt > 1)
                    iv = input(cg, { p.size() / nutt), nutt) }, &p);
                else
                    iv = input(cg, { p.size()) }, &p);
            }
            else
            {
                if (k < (m_hidden_replicates - 1) * layers)
                    iv = input(cg, m_mem_network.m_mem_dim, &p);
                else
                    iv = input(cg, m_mem_network.m_key_dim, &p);
            }
            ve.push_back(iv);
            k++;
        }

        m_mem_network.final_s() = ve;
    }

protected:
    virtual void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) {
        nutt = source.size();

        std::vector<Expression> source_embeddings;

        context.set_data_in_parallel(nutt);
        m_mem_network.set_data_in_parallel(nutt);

        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);

        encoder_fwd.start_new_sequence();

        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);

        encoder_bwd.start_new_sequence();

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        //         if (!similar_length(source))
        //         {
        //             cerr << "sentence length differs too much" << endl;
        //             abort();
        //         }
        
        src_fwd = bidirectional(slen, source, cg, p_cs, zero, encoder_fwd, encoder_bwd, hidden_dim[ENCODER_LAYER]); 
        

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)2 * hidden_dim[ENCODER_LAYER], src_len);

        /// for contet
        vector<Expression> to;
        /// take the top layer from decoder, take its final h
        to.push_back(encoder_fwd.final_h()[layers - 1]);
        to.push_back(encoder_bwd.final_h()[layers - 1]);

        Expression q_m = concatenate(to);

        Expression i_c_t = context.add_input(q_m);

        m_mem_network.add_input(i_c_t);

        cg.incremental_forward();

        vector<Expression> d_m;
        for (auto c : m_mem_network.final_c())
            d_m.push_back(c);
        for (auto h : m_mem_network.final_h())
            d_m.push_back(h);

        i_U = parameter(cg, p_U);
        src = i_U * concatenate_cols(v_src);  // precompute 

        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        decoder.start_new_sequence(d_m);  /// get the intention

        i_att_gate_A = parameter(cg, p_att_gate_A);
        i_att_gate_b = parameter(cg, p_att_gate_b);

        v_att_gate_b = concatenate_cols(vector<Expression>(nutt, i_att_gate_b));

    };

    virtual void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg, Builder* encoder_fwd, Builder* encoder_bwd, Builder* context, Builder* decoder) {
        nutt = source.size();

        std::vector<Expression> source_embeddings;

        context->set_data_in_parallel(nutt);
        m_mem_network.set_data_in_parallel(nutt);

        encoder_fwd->new_graph(cg);
        encoder_fwd->set_data_in_parallel(nutt);

        encoder_fwd->start_new_sequence();

        encoder_bwd->new_graph(cg);
        encoder_bwd->set_data_in_parallel(nutt);

        encoder_bwd->start_new_sequence();

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        //         if (!similar_length(source))
        //         {
        //             cerr << "sentence length differs too much" << endl;
        //             abort();
        //         }

        src_fwd = bidirectional(slen, source, cg, p_cs, zero, encoder_fwd, encoder_bwd, hidden_dim[ENCODER_LAYER]); 
        
        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)2 * hidden_dim[ENCODER_LAYER], src_len);

        /// for contet
        vector<Expression> to;
        /// take the top layer from decoder, take its final h
        to.push_back(encoder_fwd->final_h()[layers - 1]);
        to.push_back(encoder_bwd->final_h()[layers - 1]);

        Expression q_m = concatenate(to);

        Expression i_c_t = context->add_input(q_m);

        m_mem_network.add_input(i_c_t);

        cg.incremental_forward();

        vector<Expression> d_m;
        for (auto c : m_mem_network.final_c())
            d_m.push_back(c);
        for (auto h : m_mem_network.final_h())
            d_m.push_back(h);

        i_U = parameter(cg, p_U);
        src = i_U * concatenate_cols(v_src);  // precompute 

        decoder->new_graph(cg);
        decoder->set_data_in_parallel(nutt);
        decoder->start_new_sequence(d_m);  /// get the intention

        i_att_gate_A = parameter(cg, p_att_gate_A);
        i_att_gate_b = parameter(cg, p_att_gate_b);

        v_att_gate_b = concatenate_cols(vector<Expression>(nutt, i_att_gate_b));

    };

};

*/

template <class Builder, class Decoder>
class DynamicMemoryNetDialogue {
private:
    vector<Expression> query_obs;  /// the observed context/query
    vector<vector<Expression>> fact_encoder_state;  /// [number_of_facts][number_of_hidden_states]

    Parameters* p_fact_encoder_state;
    Parameters* p_fact_encoder_state_bias;  // for affine transformation of fact states
    Expression i_fact_encoder_state;
    Expression i_fact_encoder_state_bias;  // for affine transformation of fact states

    Parameters* p_fact_encoder_state_to_cxt;
    Parameters* p_fact_encoder_state_to_cxt_bias;  // for affine transformation of fact states to context
    Expression i_fact_encoder_state_to_cxt;
    Expression i_fact_encoder_state_to_cxt_bias;  // for affine transformation of fact states to context

    Parameters* p_cxt_to_encoder;
    Parameters* p_cxt_to_encoder_bias;  // for affine transformation of context to fact encoder
    Expression i_cxt_to_encoder;
    Expression i_cxt_to_encoder_bias;  // for affine transformation of context to fact encoder

    Parameters* p_cxt_to_decoder;
    Parameters* p_cxt_to_decoder_bias;  // for affine transformation of context to fact encoder
    Expression i_cxt_to_decoder;
    Expression i_cxt_to_decoder_bias;  // for affine transformation of context to fact encoder

protected:
    LookupParameters* p_cs;
    Parameters* p_bias;
    Parameters* p_R;  // for affine transformation after decoder
    Expression i_R, i_bias;

    vector<size_t> layers;
    Decoder decoder;  // for decoder at each turn
    Builder encoder_fwd, encoder_bwd; /// for encoder at each turn
    Builder context; // for contexter

    /// for alignment to source
    Parameters* p_U;
    Expression i_U;

    Parameters* p_va, *p_Wa;
    Parameters* p_Q;
    Expression i_Wa, i_va, i_Q;

    Parameters* p_scale;
    Expression i_scale;

    Model model;

    int vocab_size;
    int vocab_size_tgt;
    vector<unsigned> hidden_dim;
    int rep_hidden;
    int decoder_use_additional_input;

    // state variables used in the above two methods
    vector<Expression> v_src;
    Expression src;
    Expression i_sm0;  // the first input to decoder, even before observed
    std::vector<size_t> src_len;
    Expression src_fwd;
    unsigned slen;

    // for initial hidden state
    vector<Parameters*> p_h0;
    vector<Expression> i_h0;

    size_t turnid;

    unsigned int nutt; // for multiple training utterance per inibatch
    vector<cnn::real> zero;

public:
    /// for criterion
    vector<Expression> v_errs;
    size_t src_words;
    size_t tgt_words;

public:

    DynamicMemoryNetDialogue(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dim, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0)
        :   layers(layers),
            decoder(layers[DECODER_LAYER], hidden_replicates * layers[INTENTION_LAYER] * hidden_dim[INTENTION_LAYER], hidden_dim[DECODER_LAYER], &model, iscale, "decoder"),
            encoder_fwd(layers[ENCODER_LAYER], hidden_dim[ENCODER_LAYER], hidden_dim[ENCODER_LAYER], &model, iscale, "encoder_fwd"),
            encoder_bwd(layers[ENCODER_LAYER], hidden_dim[ENCODER_LAYER], hidden_dim[ENCODER_LAYER], &model, iscale, "encoder_bwd"),
            decoder_use_additional_input(decoder_use_additional_input),
            context(layers[INTENTION_LAYER], layers[ENCODER_LAYER] * hidden_replicates * hidden_dim[ENCODER_LAYER], hidden_dim[INTENTION_LAYER], &model, iscale, "context"),
            vocab_size(vocab_size_src), vocab_size_tgt(vocab_size_tgt),
            hidden_dim(hidden_dim),
            rep_hidden(hidden_replicates)
    {
        p_cs = model.add_lookup_parameters(vocab_size_src, { hidden_dim[ENCODER_LAYER] }, iscale, "p_cs");
        p_R = model.add_parameters({ vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale, "p_R");
        p_bias = model.add_parameters({ vocab_size_tgt }, iscale , "p_bias");

        p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], 2 * hidden_dim[ENCODER_LAYER] }, iscale, "p_U");

        for (size_t i = 0; i < layers[ENCODER_LAYER] * rep_hidden; i++)
        {
            p_h0.push_back(model.add_parameters({ hidden_dim[ENCODER_LAYER] }, iscale, "p_h0" + boost::lexical_cast<string>(i)));
            p_h0.back()->reset_to_zero();
        }
        zero.resize(hidden_dim[ENCODER_LAYER], 0);  /// for the no obs observation

        i_h0.clear();

        p_fact_encoder_state = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[ENCODER_LAYER] * encoder_fwd.num_h0_components() }, iscale, "p_fact_encoder_state");
        p_fact_encoder_state_bias = model.add_parameters({ hidden_dim[ENCODER_LAYER] }, iscale, "p_fact_encoder_state_bias");

        p_fact_encoder_state_to_cxt = model.add_parameters({ hidden_dim[ENCODER_LAYER] * encoder_fwd.num_h0_components(), hidden_dim[ENCODER_LAYER] }, iscale, "p_fact_encoder_state_to_cxt");
        p_fact_encoder_state_to_cxt_bias = model.add_parameters({ hidden_dim[ENCODER_LAYER] * encoder_fwd.num_h0_components() }, iscale, "p_fact_encoder_state_to_cxt_bias");

        p_cxt_to_encoder = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_dim[INTENTION_LAYER] }, iscale, "p_cxt_to_encoder");
        p_cxt_to_encoder_bias = model.add_parameters({ hidden_dim[ENCODER_LAYER] }, iscale, "p_cxt_to_encoder_bias");

        p_cxt_to_decoder = model.add_parameters({ hidden_dim[DECODER_LAYER], hidden_dim[INTENTION_LAYER] }, iscale, "p_cxt_to_decoder");
        p_cxt_to_decoder_bias = model.add_parameters({ hidden_dim[DECODER_LAYER] }, iscale, "p_cxt_to_decoder_bias");

        p_U = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[ENCODER_LAYER] }, iscale , "p_U");

        unsigned align_dim = hidden_dim[ALIGN_LAYER];
        p_Wa = model.add_parameters({ hidden_dim[ENCODER_LAYER], hidden_replicates * layers[INTENTION_LAYER] * hidden_dim[INTENTION_LAYER] }, iscale, "p_Wa");
        p_va = model.add_parameters({ hidden_dim[ENCODER_LAYER] }, 0, "p_va");
        p_Q = model.add_parameters({ hidden_dim[DECODER_LAYER], hidden_replicates * hidden_dim[DECODER_LAYER] }, iscale, "p_Q");

        p_scale = model.add_parameters({ 1 }, 0.0, "p_scale");
    }

    void assign_cxt(ComputationGraph& cg, unsigned int nutt)
    {
        throw("not implemented");
    }
    
    void assign_cxt(ComputationGraph &cg,
        const vector<int>& query)
    {
        vector<Sentence> vs(1, query);
        assign_cxt(cg, vs);
    }
    
    void assign_cxt(ComputationGraph &cg,
        const vector<vector<int>>& query)
    {
        if (query.size() == 0 || query[0].size() == 0)
        {
            return;
        }

        unsigned int nutt = query.size();
        unsigned slen;
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence();
        Expression i_query_obs = concatenate_cols(forward_directional<Builder>(slen, query, cg, p_cs, zero, encoder_fwd, hidden_dim[ENCODER_LAYER]));
        query_obs.push_back(concatenate(encoder_fwd.final_s()));

        context.new_graph(cg);
        context.set_data_in_parallel(nutt);
        context.start_new_sequence();

        context.add_input(query_obs.back());
        vector<Expression> i_cxt_final_s;
        vector<Expression> fwd_processed;
        vector<Expression> prv_fwd_processed;
        for (auto p : fact_encoder_state)
        {
            Expression i_fc = i_fact_encoder_state * concatenate(p) + i_fact_encoder_state_bias;
            fwd_processed.push_back(i_fc);
        }
        vector<size_t> each_fact_len;
        each_fact_len.push_back(fact_encoder_state.size());

        for (size_t dp = 0; dp < REASONING_STEPS; dp++)
        {
            i_cxt_final_s = context.final_s();

            /// encoder update
            if (fwd_processed.size() == 0)
            {
                vector<Expression> v_cxt2enc;
                for (auto p : i_cxt_final_s)
                    v_cxt2enc.push_back(i_cxt_to_encoder * p + i_cxt_to_encoder_bias);
                encoder_fwd.new_graph(cg);
                encoder_fwd.set_data_in_parallel(nutt);
                encoder_fwd.start_new_sequence(v_cxt2enc);


                for (auto p : prv_fwd_processed)
                {
                    fwd_processed.push_back(encoder_fwd.add_input(p));
                }
            }

            vector<Expression>
                v_facts = shuffle_data(concatenate_cols(fwd_processed), (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], each_fact_len);

            Expression i_h_tm1 = concatenate(i_cxt_final_s);
            vector<Expression> alpha;
            vector<Expression> v_obs = attention_using_bilinear(v_facts, each_fact_len, i_Wa, i_h_tm1, hidden_dim[ENCODER_LAYER], nutt, alpha, i_scale);
            if (verbose)
            {
                display_value(concatenate_cols(alpha), cg, "attention weights");
                display_value(i_scale, cg, "exponential coeff");
            }
            Expression cxt_input = concatenate_cols(v_obs);
            context.add_input(i_fact_encoder_state_to_cxt * cxt_input + i_fact_encoder_state_to_cxt_bias);

            prv_fwd_processed = fwd_processed;
            fwd_processed.clear();
        }
    }

    void start_new_single_instance(const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        start_new_instance(source, cg);
    }

    void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (source.size() == 0 || source[0].size() == 0)
            return;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);
            i_Q = parameter(cg, p_Q);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_fact_encoder_state = parameter(cg, p_fact_encoder_state);
            i_fact_encoder_state_bias = parameter(cg, p_fact_encoder_state_bias);
            i_fact_encoder_state_to_cxt = parameter(cg, p_fact_encoder_state_to_cxt);
            i_fact_encoder_state_to_cxt_bias = parameter(cg, p_fact_encoder_state_to_cxt_bias);
            i_cxt_to_encoder = parameter(cg, p_cxt_to_encoder);
            i_cxt_to_encoder_bias = parameter(cg, p_cxt_to_encoder_bias);
            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_cxt_to_decoder_bias = parameter(cg, p_cxt_to_decoder_bias);

            i_scale = exp(parameter(cg, p_scale));
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the reverse direction to encoder source side
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        if (fact_encoder_state.size() > 0)
            encoder_fwd.start_new_sequence(fact_encoder_state.back());
        else
            encoder_fwd.start_new_sequence(i_h0);

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, encoder_fwd, hidden_dim[ENCODER_LAYER]));

        /// have input to context RNN
        vector<Expression> to = encoder_fwd.final_s();
        fact_encoder_state.push_back(to);
    }

    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg)
    {
        vector<Expression> exp;
        throw("not implemented");
        return exp;
    }

    vector<Expression> build_comp_graph(const std::vector<std::vector<int>> &source,
        const std::vector<std::vector<int>>& osent, 
        ComputationGraph &cg)
    {
        // decoder
        vector<Expression> errs;
        unsigned int nutt;
        start_new_instance(source, cg);

        if (osent.size() == 0 || osent[0].size() == 0)
            return errs; 

        nutt = osent.size();

        int oslen = 0;
        for (auto p : osent)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        Expression i_bias_mb = concatenate_cols(vector<Expression>(nutt, i_bias));

        decoder.new_graph(cg); 
        decoder.set_data_in_parallel(nutt); 
        vector<Expression> v_to_dec;
        for (auto p : context.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p + i_cxt_to_decoder_bias);
        decoder.start_new_sequence(v_to_dec);

        /// the task is classification
        assert(oslen == 1);

        Expression i_frm_cxt = concatenate(context.final_s());

        Expression i_y_t = decoder.add_input(i_frm_cxt);
        Expression i_r_t = i_bias_mb + i_R * i_y_t;
        if (verbose)
        {
            display_value(i_r_t, cg, "output before softmax");
            cout << "reference: ";
            for (auto p : osent)
            {
                cout << p[0] << " ";
            }
            cout << endl;
        }

        Expression x_r_t = reshape(i_r_t, { vocab_size_tgt * nutt });
        for (size_t i = 0; i < nutt; i++)
        {
            /// only compute errors on with output labels
            Expression r_r_t = pickrange(x_r_t, i * vocab_size_tgt, (i + 1)*vocab_size_tgt);
            Expression i_ydist = log_softmax(r_r_t);
            errs.push_back(pick(i_ydist, osent[i][0]));

            tgt_words++;
        }

        Expression i_nerr = -sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return v_errs;
    };

    std::vector<int> decode(const Sentence&source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        vector<int> vres;
        throw("not implemented");
        return vres;
    }

    std::vector<Sentence> batch_decode(const vector<Sentence>&source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        vector<Sentence> vres;
        throw("not implemented");
        return vres;
    }

    std::vector<int> decode_tuple(const SentenceTuple&source, ComputationGraph& cg, cnn::Dict  &sdict, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;

        int t = 0;

        start_new_single_instance(source.first, cg);

        if (source.last.size() == 0)  /// no query
            return target;

        assign_cxt(cg, source.last);

        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : context.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p + i_cxt_to_decoder_bias);
        decoder.start_new_sequence(v_to_dec);

        target.push_back(sos_sym);

        Expression i_y_t = decoder.add_input(concatenate(context.final_s())); 
        Expression i_r_t = i_bias + i_R * i_y_t;
        Expression ydist = softmax(i_r_t);

        // find the argmax next word (greedy)
        unsigned w = 0;
        auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
        auto pr_w = dist[w];
        for (unsigned x = 1; x < dist.size(); ++x) {
            if (dist[x] > pr_w) {
                w = x;
                pr_w = dist[x];
            }
            target.push_back(w);
        }

        turnid++;

        return target;
    }

    void serialise_context(ComputationGraph& cg)
    {
    }

    /// for context
    void reset()
    {
        turnid = 0;

        i_h0.clear();

        v_errs.clear();
        src_words = 0;
        tgt_words = 0;

        query_obs.clear();
        fact_encoder_state.clear();
    }

};

/**
use simple models 
encoder:
prev_response -> vector representation -> 
current_user  -> vector representation -> 
                 previous intention hidden -> combine at each layer linearly (z_k = c z_k-1 + W_p y_k-1 + W_u x_k)
decoder:
simple RNN, with initial state from the above combination at each layer. 
or can be simpler such as bilinear model
*/
template <class Builder, class Decoder>
class MultiSource_LinearEncoder : public DialogueBuilder<Builder, Decoder>{
protected:
    /// time-dependent embedding weight
    map<size_t, map<size_t, tExpression>> m_time_embedding_weight;
    Expression i_zero;
    Builder combiner; /// the combiner that combines the multipe sources of inputs, and possibly its history

    Parameters * p_cxt_to_decoder, *p_enc_to_intention;
    Expression i_cxt_to_decoder, i_enc_to_intention;
public:
    MultiSource_LinearEncoder(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :DialogueBuilder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale),
        combiner(layers[INTENTION_LAYER], vector<unsigned>{hidden_dims[INTENTION_LAYER], hidden_dims[INTENTION_LAYER], hidden_dims[INTENTION_LAYER]}, &model, iscale)
    {
        p_cxt_to_decoder = model.add_parameters({ hidden_dim[DECODER_LAYER], hidden_dim[INTENTION_LAYER] }, iscale, "p_cxt_to_decoder");

        p_enc_to_intention= model.add_parameters({ hidden_dim[INTENTION_LAYER], hidden_dim[ENCODER_LAYER]}, iscale, "p_enc_to_intention");
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg) 
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::start_new_instance" << endl;
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;
        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);
            if (verbose)
                display_value(i_zero,  cg, "i_zero");
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            if (verbose)
                cout << "has prv_response" << endl;
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
            if (verbose)
                cout << "done encode prv_response" << endl;
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }
        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(slen, source, cg, p_cs));
        if (verbose)
            cout << "done concatenate_cols(average_embedding(slen, source, cg, p_cs))" << endl;

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);
        if (verbose)
            cout << "done encode current input" << endl;

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);
        if (verbose)
            cout << "done combiner add_input" << endl;

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention
        if (verbose)
            cout << "done decoder.start_new_sequence" << endl;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::decoder_step" << endl;
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        if (verbose)
            cout << "decoder_step" << endl;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
            {
                if (verbose)
                {
                    cout << "to call lookup of " << p << endl;
                }
                i_x_x = lookup(cg, p_cs, p);
                if (verbose)
                    cout << "done call lookup " << p << endl;
            }
            else
            {
                if (verbose)
                    cout << "to call i_zero" << endl;
                i_x_x = i_zero;
            }
            if (verbose)
                cout << " before calling display_value i_x_x" << endl;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        if (verbose)
            cout << "to call concatenate_cols" << endl;
        Expression i_obs = concatenate_cols(v_x_t);
        if (verbose)
            display_value(i_obs, cg, "i_obs");

        if (verbose)
            cout << "to call decoder.add_input" << endl;
        i_h_t = decoder.add_input(i_obs);
        if (verbose)
            display_value(i_h_t, cg, "i_h_t");

        return i_h_t;
    }

    vector<Expression> build_graph(
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        vector<vector<int>> prv_response;
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs;
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            if (verbose)
                display_value(i_y_t, cg, "i_y_t");
            Expression i_r_t = i_R * i_y_t;
            if (verbose)
            {
                cg.incremental_forward(); 
                cout << "i_r_t" << endl;
            }

            Expression x_r_t = reshape(i_r_t, { vocab_size * nutt });
            if (verbose)
            {
                cg.incremental_forward();
                cout << "x_r_t" << endl;
            }
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "r_r_t" << endl;
                    }
                    Expression i_ydist = log_softmax(r_r_t);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "i_ydist" << endl;
                    }
                    this_errs[i].push_back(-pick(i_ydist, target_response[i][t + 1]));
                    if (verbose)
                        display_value(this_errs[i].back(), cg, "this_errs");
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : v_decoder.back()->final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input, 
        const std::vector<std::vector<int>>& target_response, 
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "MultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            if (verbose)
                display_value(i_y_t, cg, "i_y_t");
            Expression i_r_t = i_R * i_y_t;
            if (verbose)
            {
                cg.incremental_forward();
                cout << "i_r_t" << endl;
            }

            Expression x_r_t = reshape(i_r_t, { vocab_size * nutt });
            if (verbose)
            {
                cg.incremental_forward();
                cout << "x_r_t" << endl;
            }
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "r_r_t" << endl;
                    }
                    Expression i_ydist = log_softmax(r_r_t);
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "i_ydist" << endl;
                    }
                    this_errs[i].push_back(-pick(i_ydist, target_response[i][t + 1]));
                    if (verbose)
                        display_value(this_errs[i].back(), cg, "this_errs utt[" + boost::lexical_cast<string>(i) + "]");
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    if (verbose)
                    {
                        cg.incremental_forward();
                        cout << "before looping decodre.final_s()" << endl;
                    }
                    
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    if (verbose)
                        display_value(v_t.back(), cg, "v_t");
                    v_decoder_context[i] = v_t;
                    if (verbose)
                    {
                        cout << "v_t pushed" << endl;
                    }
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);
        if (verbose)
            display_value(i_nerr, cg, "i_nerr");
        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response; 

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    /**
    1) save context hidden state
    in last_cxt_s as [replicate_hidden_layers][nutt]
    2) organize the context from decoder
    data is organized in v_decoder_context as [nutt][replicate_hidden_layers]
    after this process, last_decoder_s will save data in dimension [replicate_hidden_layers][nutt]
    */
    void serialise_context(ComputationGraph& cg)
    {
        /// get the top output
        serialise(cg, combiner);

        vector<vector<cnn::real>> v_last_d;
        unsigned int nutt = v_decoder_context.size();
        size_t ndim = v_decoder_context[0].size();
        v_last_d.resize(ndim);

        size_t ik = 0;
        vector<vector<cnn::real>> vm;
        for (const auto &p : v_decoder_context)
        {
            /// for each utt
            vm.clear();
            for (const auto &v : p)
                vm.push_back(get_value(v, cg));

            size_t iv = 0;
            for (auto p : vm)
            {
                if (ik == 0)
                {
                    v_last_d[iv].resize(nutt * p.size());
                }
                std::copy_n(p.begin(), p.size(), &v_last_d[iv][ik * p.size()]);
                iv++;
            }
            ik++;
        }
        last_decoder_s = v_last_d;
    }
};

/** additionally with attention
*/
template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder : public MultiSource_LinearEncoder <Builder, Decoder>{
protected:
    /// for location prediction for local attention
    Parameters* p_va_local, *p_Wa_local, *p_ba_local;
    Expression i_va_local, i_Wa_local, i_ba_local;

    /// for the expoential scaling of the softmax
    Parameters* p_scale;
    Expression i_scale;

    Parameters* p_Wa, *p_va;
    Expression i_Wa, i_va;

    /// a single layer MLP
    DNNBuilder attention_layer;
    vector<Expression> attention_output_for_this_turn; /// [number of turn]

    Expression i_zero;
public:
    AttMultiSource_LinearEncoder(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :MultiSource_LinearEncoder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale) ,
        attention_layer(1, vector<unsigned>{hidden_dim[ENCODER_LAYER] + hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER], hidden_dim[DECODER_LAYER]}, &model, iscale, "attention_layer")
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder" << endl;
        
        p_Wa_local = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[DECODER_LAYER] }, iscale);
        p_ba_local = model.add_parameters({ hidden_dim[ALIGN_LAYER] }, iscale);
        p_va_local = model.add_parameters({ 1, hidden_dim[ALIGN_LAYER] }, iscale);

        unsigned int align_dim = hidden_dim[ALIGN_LAYER];
        p_Wa = model.add_parameters({ align_dim, hidden_dim[DECODER_LAYER] }, iscale);
        p_va = model.add_parameters({ align_dim }, iscale);

        p_scale = model.add_parameters({ 1 }, 0.0);

    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg) 
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            attention_output_for_this_turn.clear();

            i_Wa_local = parameter(cg, p_Wa_local);
            i_va_local = parameter(cg, p_va_local);
            i_ba_local = parameter(cg, p_ba_local);

            i_scale = exp(parameter(cg, p_scale));

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the representation of inputs
        Expression src_tmp = concatenate_cols(embedding(slen, source, cg, p_cs, zero, hidden_dim[ENCODER_LAYER]));
        v_src = shuffle_data(src_tmp, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);

        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(slen, source, cg, p_cs));

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        if (verbose)
            cout << "AttMultiSource_LinearEncoder::decoder_step" << endl;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        vector<Expression> alpha;
        vector<Expression> position = local_attention_to(cg, src_len, i_Wa_local, i_ba_local, i_va_local, i_h_t, nutt);
        if (verbose)
        {
            for (auto &p : position){
                display_value(p, cg, "predicted local attention position ");
            }
        }

        vector<Expression> v_context_to_source = attention_using_bilinear_with_local_attention(v_src, src_len, i_Wa, i_h_t, hidden_dim[ENCODER_LAYER], nutt, alpha, i_scale, position);
        if (verbose)
        {
            size_t k = 0;
            for (auto &p : alpha){
                display_value(p, cg, "attention_to_source_weight_" + boost::lexical_cast<string>(k++));
            }
        }

        /// compute attention
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenate_cols(v_context_to_source) });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        return i_h_attention_t;
    }
    
    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_R * i_y_t;
            Expression i_ydist = log_softmax(i_r_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size; 
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_R * i_y_t;
            Expression i_ydist = log_softmax(i_r_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size; 
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression i_r_t = i_R * i_y_t;
            Expression ydist = softmax(i_r_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }
};

/**
Use attentional max-entropy features or direct features
*/
template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder_WithMaxEntropyFeature : public AttMultiSource_LinearEncoder <Builder, Decoder>{
protected:
    cnn::real r_interpolate_between_response_and_direct_feature; /// scale to 
    cnn::real r_softmax_scale; /// for attention softmax exponential scale
public:
    AttMultiSource_LinearEncoder_WithMaxEntropyFeature(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :AttMultiSource_LinearEncoder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale)
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder_WithMaxEntropyFeature" << endl;

        r_interpolate_between_response_and_direct_feature = 0.7;
        r_softmax_scale = 1.0;

        unsigned int align_dim = hidden_dim[ALIGN_LAYER];
        p_Wa = model.add_parameters({ align_dim, hidden_dim[DECODER_LAYER] }, iscale, "p_Wa");
        p_va = model.add_parameters({ align_dim }, iscale, "p_va");
        p_U  = model.add_parameters({ hidden_dim[ALIGN_LAYER], hidden_dim[ENCODER_LAYER] }, iscale, "p_U");
    }

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_U = parameter(cg, p_U);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            attention_output_for_this_turn.clear();

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the representation of inputs
        v_src = embedding_spkfirst(source, cg, p_cs, hidden_dim[ENCODER_LAYER]); 
        src = i_U * concatenate_cols(v_src);

        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(src_len, hidden_dim[ENCODER_LAYER], v_src));

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention

    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        if (verbose)
            cout << "AttMultiSource_LinearEncoder::decoder_step" << endl;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        /// return the source side representation at output position after attention
        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

        /// compute response
        Expression concatenated_src = concatenate_cols(v_context_to_source);
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src});
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        Expression i_output = i_R * (r_interpolate_between_response_and_direct_feature * i_h_attention_t + (1 - r_interpolate_between_response_and_direct_feature)* concatenated_src);

        return i_output;
    }

    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }
};

/**
Use global max-entropy and attentional direct features 
*/
template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature : public AttMultiSource_LinearEncoder_WithMaxEntropyFeature<Builder, Decoder>{
protected:
    LookupParameters* p_tgt_side_emb;  /// target side embedding
    Parameters * p_global_me_weight;/// weight for global ME feature
    Expression i_global_me_weight;

    Expression i_glb_me_feature;     /// computed global me feature

public:
    AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :AttMultiSource_LinearEncoder_WithMaxEntropyFeature(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale)
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature" << endl;

        /// have bigram history to have trigram ME feature
        p_tgt_side_emb = model.add_lookup_parameters(vocab_size_src, { hidden_dim[DECODER_LAYER] }, iscale);
        p_global_me_weight = model.add_parameters({vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale);
    }

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_U = parameter(cg, p_U);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            i_global_me_weight = parameter(cg, p_global_me_weight);

            attention_output_for_this_turn.clear();

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the representation of inputs
        v_src = embedding_spkfirst(source, cg, p_cs, hidden_dim[ENCODER_LAYER]);
        src = i_U * concatenate_cols(v_src);

        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(src_len, hidden_dim[ENCODER_LAYER], v_src));

        ////// get the global ME feature  //////
        /// first get the representation of inputs
        vector<Expression> v_target_side_emb = embedding_spkfirst(source, cg, p_tgt_side_emb, hidden_dim[DECODER_LAYER]);
        vector<Expression> v_ave_global_me_raw = average_embedding(src_len, hidden_dim[DECODER_LAYER], v_target_side_emb);
        i_glb_me_feature = i_global_me_weight * concatenate_cols(v_ave_global_me_raw);

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention

    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        if (verbose)
            cout << "AttMultiSource_LinearEncoder::decoder_step" << endl;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        /// return the source side representation at output position after attention
        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

        /// compute response
        Expression concatenated_src = concatenate_cols(v_context_to_source);
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        cnn::real me_feature_weight = (1 - r_interpolate_between_response_and_direct_feature) / 2;
        Expression i_output = i_R * (r_interpolate_between_response_and_direct_feature * i_h_attention_t 
            + me_feature_weight * concatenated_src) 
            + me_feature_weight * i_glb_me_feature;

        return i_output;
    }
    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

};

/**
Use global max-entropy and attentional direct features
To speed-up, each source has been padded with </s> to have equal length of source sentences
*/
template <class Builder, class Decoder>
class AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature_Batch : public AttMultiSource_LinearEncoder_WithMaxEntropyFeature<Builder, Decoder>{
protected:
    LookupParameters* p_tgt_side_emb;  /// target side embedding
    Parameters * p_global_me_weight;/// weight for global ME feature
    Expression i_global_me_weight;

    Expression i_glb_me_feature;     /// computed global me feature

public:
    AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature_Batch(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :AttMultiSource_LinearEncoder_WithMaxEntropyFeature(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale)
    {
        if (verbose)
            cout << "start AttMultiSource_LinearEncoder_WithMaxEntropyFeature_AndGlobalDirectFeature" << endl;

        /// have bigram history to have trigram ME feature
        p_tgt_side_emb = model.add_lookup_parameters(vocab_size_src, { hidden_dim[DECODER_LAYER] }, iscale);
        p_global_me_weight = model.add_parameters({ vocab_size_tgt, hidden_dim[DECODER_LAYER] }, iscale);
    }

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (verbose)
            cout << "start_new_instance" << endl;

        if (i_h0.size() == 0)
        {
            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_U = parameter(cg, p_U);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            i_global_me_weight = parameter(cg, p_global_me_weight);

            attention_output_for_this_turn.clear();

            i_Wa = parameter(cg, p_Wa);
            i_va = parameter(cg, p_va);

            attention_layer.new_graph(cg);
            attention_layer.set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the representation of inputs
        v_src = embedding_spkfirst(source, cg, p_cs, hidden_dim[ENCODER_LAYER]);
        src = i_U * concatenate_cols(v_src);

        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(src_len, hidden_dim[ENCODER_LAYER], v_src));

        ////// get the global ME feature  //////
        /// first get the representation of inputs
        vector<Expression> v_target_side_emb = embedding_spkfirst(source, cg, p_tgt_side_emb, hidden_dim[DECODER_LAYER]);
        vector<Expression> v_ave_global_me_raw = average_embedding(src_len, hidden_dim[DECODER_LAYER], v_target_side_emb);
        i_glb_me_feature = i_global_me_weight * concatenate_cols(v_ave_global_me_raw);

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention

    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
    {
        Expression i_c_t;
        unsigned int nutt = trg_tok.size();
        Expression i_h_t;  /// the decoder output before attention
        Expression i_h_attention_t; /// the attention state
        vector<Expression> v_x_t;

        if (verbose)
            cout << "AttMultiSource_LinearEncoder::decoder_step" << endl;
        for (auto p : trg_tok)
        {
            Expression i_x_x;
            if (p >= 0)
                i_x_x = lookup(cg, p_cs, p);
            else
                i_x_x = i_zero;
            if (verbose)
                display_value(i_x_x, cg, "i_x_x");
            v_x_t.push_back(i_x_x);
        }

        Expression i_obs = concatenate_cols(v_x_t);
        Expression i_input;
        if (attention_output_for_this_turn.size() <= turnid)
        {
            i_input = concatenate({ i_obs, concatenate_cols(vector<Expression>(nutt, i_zero)) });
        }
        else
        {
            i_input = concatenate({ i_obs, attention_output_for_this_turn.back() });
        }

        i_h_t = decoder.add_input(i_input);

        /// return the source side representation at output position after attention
        vector<Expression> alpha;
        vector<Expression> v_context_to_source = attention_to_source_batch(v_src, src_len, i_va, i_Wa, i_h_t, src, hidden_dim[ALIGN_LAYER], nutt, alpha, r_softmax_scale);

        /// compute response
        Expression concatenated_src = concatenate_cols(v_context_to_source);
        Expression i_combined_input_to_attention = concatenate({ i_h_t, concatenated_src });
        i_h_attention_t = attention_layer.add_input(i_combined_input_to_attention);

        if (attention_output_for_this_turn.size() <= turnid)
            attention_output_for_this_turn.push_back(i_h_attention_t);
        else
            /// refresh the attention output for this turn
            attention_output_for_this_turn[turnid] = i_h_attention_t;

        cnn::real me_feature_weight = (1 - r_interpolate_between_response_and_direct_feature) / 2;
        Expression i_output = i_R * (r_interpolate_between_response_and_direct_feature * i_h_attention_t
            + me_feature_weight * concatenated_src)
            + me_feature_weight * i_glb_me_feature;

        return i_output;
    }
    vector<Expression> build_graph(const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                long offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        if (verbose)
            cout << "AttMultiSource_LinearEncoder::build_graph" << endl;
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_ydist = log_softmax(i_y_t);

            Expression x_r_t = reshape(i_ydist, { vocab_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                int offset = i * vocab_size;
                if (t < target_response[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    this_errs[i].push_back(-pick(x_r_t, target_response[i][t + 1] + offset));
                    tgt_words++;
                }
                else if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);

        for (auto &p : this_errs)
            errs.push_back(sum(p));
        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);
            Expression ydist = softmax(i_y_t);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = get_value(ydist, cg); // evaluates last expression, i.e., ydist
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);
        serialise_context(cg);

        turnid++;
        return target;
    }

};

template <class Builder, class Decoder>
class ClsBasedMultiSource_LinearEncoder : public MultiSource_LinearEncoder <Builder, Decoder>{
protected:
    ClsBasedBuilder* p_clsbased_error;
    cnn::real iscale; 
    vector<int> hash; 

public:
    ClsBasedMultiSource_LinearEncoder(Model& model,
        unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
        const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :MultiSource_LinearEncoder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale) , iscale(iscale)
    {
        p_clsbased_error = nullptr;
    }

    void start_new_instance(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &source,
        ComputationGraph &cg)
    {
        nutt = source.size();
        std::vector<Expression> v_tgt2enc;

        if (i_h0.size() == 0)
        {
            i_h0.clear();
            for (auto p : p_h0)
            {
                i_h0.push_back(concatenate_cols(vector<Expression>(nutt, parameter(cg, p))));
            }

            combiner.new_graph(cg);

            if (last_context_exp.size() == 0)
                combiner.start_new_sequence();
            else
                combiner.start_new_sequence(last_context_exp);
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            i_zero = input(cg, { (hidden_dim[DECODER_LAYER]) }, &zero);

            p_clsbased_error->new_graph(cg);
            p_clsbased_error->set_data_in_parallel(nutt);
        }

        std::vector<Expression> source_embeddings;
        std::vector<Expression> v_last_decoder_state;

        /// take the previous response as input
        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence(i_h0);
        if (prv_response.size() > 0)
        {
            /// get the raw encodeing from source
            unsigned int prvslen = 0;
            Expression prv_response_enc = concatenate_cols(average_embedding(prvslen, prv_response, cg, p_cs));
            encoder_fwd.add_input(prv_response_enc);
        }

        /// encode the source side input, with intial state from the previous response
        /// this is a way to combine the previous response and the current input
        /// notice that for just one run of the RNN, 
        /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence(encoder_fwd.final_s());

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }
        /// get the raw encodeing from source
        src_fwd = concatenate_cols(average_embedding(slen, source, cg, p_cs));

        /// combine the previous response and the current input by adding the current input to the 
        /// encoder that is initialized from the state of the encoder for the previous response
        encoder_bwd.add_input(src_fwd);

        /// update intention, with inputs for each each layer for combination
        /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
        vector<Expression> v_to_intention;
        for (auto p : encoder_bwd.final_s())
            v_to_intention.push_back(i_enc_to_intention* p);
        combiner.add_input(v_to_intention);

        /// decoder start with a context from intention 
        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        vector<Expression> v_to_dec;
        for (auto p : combiner.final_s())
            v_to_dec.push_back(i_cxt_to_decoder * p);
        decoder.start_new_sequence(v_to_dec);  /// get the intention
    }

    bool load_cls_info_from_file(string word2clsfn, string clsszefn, Dict& sd, Model& model)
    {
        vector<int> clssize;
        vector<long> word2cls;
        vector<long> dict_wrd_id2within_class_id;

        ClsBasedBuilder * p_tmp_clsbased  = new ClsBasedBuilder();
        p_tmp_clsbased->load_word2cls_fn(word2clsfn, sd, word2cls, dict_wrd_id2within_class_id, clssize);
        delete p_tmp_clsbased;

        p_clsbased_error = new ClsBasedBuilder(hidden_dim[DECODER_LAYER], clssize, word2cls, dict_wrd_id2within_class_id, model, iscale, "class-based scoring");
        return true;
    }

    ~ClsBasedMultiSource_LinearEncoder()
    {
        delete[] p_clsbased_error;
    }

    vector<Expression> build_graph(
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        unsigned int nutt = current_user_input.size();
        vector<vector<int>> prv_response;
        start_new_instance(prv_response, current_user_input, cg);

        vector<Expression> errs;

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            vector<long> vtarget;
            for (auto p : target_response)
            {
                if (t + 1< p.size())
                    vtarget.push_back(p[t + 1]);
                else
                    vtarget.push_back(-1);
            }
            errs.push_back(sum(p_clsbased_error->add_input(i_y_t, vtarget)));

            for (size_t i = 0; i < nutt; i++)
            {
                if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &prv_response,
        const std::vector<std::vector<int>> &current_user_input,
        const std::vector<std::vector<int>>& target_response,
        ComputationGraph &cg)
    {
        unsigned int nutt = current_user_input.size();
        start_new_instance(prv_response, current_user_input, cg);

        vector<Expression> errs;

        int oslen = 0;
        for (auto p : target_response)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        v_decoder_context.clear();
        v_decoder_context.resize(nutt);
        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : target_response)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);

            vector<long> vtarget;
            for (auto p : target_response)
            {
                if (t + 1< p.size())
                    vtarget.push_back(p[t + 1]);
                else
                    vtarget.push_back(-1);
            }
            vector<Expression> i_r_t = p_clsbased_error->add_input(i_y_t, vtarget);
            if (i_r_t.size() > 0)
                errs.push_back(sum(i_r_t));
            tgt_words += i_r_t.size();

            for (size_t i = 0; i < nutt; i++)
            {
                if (t == target_response[i].size() - 1)
                {
                    /// get the last hidden state to decode the i-th utterance
                    vector<Expression> v_t;
                    for (auto p : decoder.final_s())
                    {
                        Expression i_tt = reshape(p, { (nutt * hidden_dim[DECODER_LAYER]) });
                        int stt = i * hidden_dim[DECODER_LAYER];
                        int stp = stt + hidden_dim[DECODER_LAYER];
                        Expression i_t = pickrange(i_tt, stt, stp);
                        v_t.push_back(i_t);
                    }
                    v_decoder_context[i] = v_t;
                }
            }
        }

        save_context(cg);
        serialise_context(cg);

        Expression i_nerr = sum(errs);

        v_errs.push_back(i_nerr);
        turnid++;
        return errs;
    };

    void start_new_single_instance(const std::vector<int> &prv_response, const std::vector<int> &src, ComputationGraph &cg)
    {
        std::vector<std::vector<int>> source(1, src);
        std::vector<std::vector<int>> prv_resp;
        if (prv_response.size() > 0)
            prv_resp.resize(1, prv_response);
        start_new_instance(prv_resp, source, cg);
    }

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;
        Sentence prv_response;

        start_new_single_instance(prv_response, source, cg);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = p_clsbased_error->respond(i_y_t, cg); 
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

    std::vector<int> decode(const std::vector<int> &prv_response, const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);
        int t = 0;

        start_new_single_instance(prv_response, source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();

        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_single_instance_step(target.back(), cg);

            // find the argmax next word (greedy)
            unsigned w = 0;
            auto dist = p_clsbased_error->respond(i_y_t, cg);
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        v_decoder_context.push_back(decoder.final_s());
        save_context(cg);

        turnid++;
        return target;
    }

};


}; // namespace cnn
