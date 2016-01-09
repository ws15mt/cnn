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
#include "ext/dialogue/dialogue.h"

#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

using namespace std;
namespace cnn {

template<class Builder, class Decoder>
class CxtEncDecModel : public DialogueBuilder<Builder, Decoder>{
private :
    Expression i_R;

public:
    CxtEncDecModel(cnn::Model& model, int vocab_size_src, int vocab_size_tgt, const vector<size_t>& layers, const vector<unsigned>& hidden_dims, int hidden_replicates, int decoder_use_additional_input = 0, int mem_slots = 0, cnn::real iscale = 1.0) :
        DialogueBuilder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, decoder_use_additional_input, mem_slots, iscale)
    {
    }

public:

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

            i_cxt2dec_w = parameter(cg, p_cxt2dec_w);
            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);
        }

        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence();

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the backward direction encoding of the source
        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, encoder_bwd, hidden_dim[ENCODER_LAYER]));

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);

        /// for contet
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
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg){
        size_t nutt = source.size();
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
                if (t < p.size())
                    vobs.push_back(p[t]);
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
                    this_errs[i].push_back( -pick(i_ydist, osent[i][t + 1]));
                    tgt_words++;
                }
            }
        }

        turnid++;
        for (auto &p : this_errs)
            errs.push_back(sum(p));
        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);

        //  std::cerr << tdict.Convert(target.back());
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        turnid++;

        return target;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
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
                i_x_x = input(cg, { hidden_dim[DECODER_LAYER] }, &zero);
            v_x_t.push_back(i_x_x);
        }
        i_x_t = concatenate_cols(v_x_t);

        Expression i_y_t = decoder.add_input(i_x_t);

        return i_y_t;
    };

};

/** sequence to sequence encoder decodr 
*/
template<class Builder, class Decoder>
class Seq2SeqEncDecModel : public DialogueBuilder<Builder, Decoder>{
private:
    Expression i_R;

public:
    Seq2SeqEncDecModel(cnn::Model& model, int vocab_size_src, int vocab_size_tgt, const vector<size_t>& layers, const vector<unsigned>& hidden_dims, int hidden_replicates, int decoder_use_additional_input = 0, int mem_slots = 0, cnn::real iscale = 1.0) :
        DialogueBuilder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, decoder_use_additional_input, mem_slots, iscale)
    {
    }

public:

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

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
        }

        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence();

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        for (auto p : src_len)
        {
            src_words += (p - 1);
        }

        /// get the backward direction encoding of the source
        src_fwd = concatenate_cols(backward_directional<Builder>(slen, source, cg, p_cs, zero, encoder_bwd, hidden_dim[ENCODER_LAYER]));

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)hidden_dim[ENCODER_LAYER], src_len);

        /// for contet
        /// have input to context RNN
        vector<Expression> to = encoder_bwd.final_s();

        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        decoder.start_new_sequence(to);  /// get the intention
    };

    vector<Expression> build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg){
        size_t nutt = source.size();
        start_new_instance(source, cg);

        vector<vector<Expression>> this_errs(nutt);
        vector<Expression> errs;

        this_errs.resize(nutt);
        int oslen = 0;
        for (auto p : osent)
            oslen = (oslen < p.size()) ? p.size() : oslen;

        for (int t = 0; t < oslen; ++t) {
            vector<int> vobs;
            for (auto p : osent)
            {
                if (t < p.size())
                    vobs.push_back(p[t]);
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
                    this_errs[i].push_back( -pick(i_ydist, osent[i][t + 1]));
                    tgt_words++;
                }
            }
        }

        turnid++;

        for (auto &p : this_errs)
            errs.push_back(sum(p));

        return errs;
    };

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);

        //  std::cerr << tdict.Convert(target.back());
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

            //        std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
            t += 1;
            target.push_back(w);
        }

        turnid++;

        return target;
    }

    Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
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
                i_x_x = input(cg, { hidden_dim[DECODER_LAYER] }, &zero);
            v_x_t.push_back(i_x_x);
        }
        i_x_t = concatenate_cols(v_x_t);

        Expression i_y_t = decoder.add_input(i_x_t);

        return i_y_t;
    };

};


}; // namespace cnn
