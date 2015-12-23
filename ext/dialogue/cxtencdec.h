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
public:
    CxtEncDecModel(cnn::Model& model, int vocab_size_src, int vocab_size_tgt, const vector<size_t>& layers, const vector<unsigned>& hidden_dims, int hidden_replicates, int decoder_use_additional_input = 0, int mem_slots = 0, cnn::real iscale = 1.0) :
        DialogueBuilder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, decoder_use_additional_input, mem_slots, iscale)
    {}

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
            context.set_data_in_parallel(nutt);
        }

        encoder_fwd.new_graph(cg);
        encoder_fwd.set_data_in_parallel(nutt);
        encoder_fwd.start_new_sequence();

        encoder_bwd.new_graph(cg);
        encoder_bwd.set_data_in_parallel(nutt);
        encoder_bwd.start_new_sequence();

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        src_fwd = bidirectional<Builder>(slen, source, cg, p_cs, zero, encoder_fwd, encoder_bwd, HIDDEN_DIM);

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)2 * hidden_dim[ENCODER_LAYER], src_len);

        /// for contet
        vector<Expression> to;
        /// take the top layer from decoder, take its final h
        to.push_back(encoder_fwd.final_h()[layers[ENCODER_LAYER] - 1]);
        to.push_back(encoder_bwd.final_h()[layers[ENCODER_LAYER] - 1]);

        Expression q_m = concatenate(to);

        context.add_input(q_m);

        vector<Expression> d_m = context.final_s();

        decoder.new_graph(cg);
        decoder.set_data_in_parallel(nutt);
        decoder.start_new_sequence(d_m);  /// get the intention
    };

    Expression build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg){
        size_t nutt;
        start_new_instance(source, cg);

        // decoder
        vector<Expression> errs;

        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias

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
                if (t < p.size())
                    vobs.push_back(p[t]);
                else
                    vobs.push_back(-1);
            }
            Expression i_y_t = decoder_step(vobs, cg);
            Expression i_r_t = i_bias_mb + i_R * i_y_t;

            Expression x_r_t = reshape(i_r_t, { (long)vocab_size * (long)nutt });
            for (size_t i = 0; i < nutt; i++)
            {
                if (t < osent[i].size() - 1)
                {
                    /// only compute errors on with output labels
                    Expression r_r_t = pickrange(x_r_t, i * vocab_size, (i + 1)*vocab_size);
                    Expression i_ydist = log_softmax(r_r_t);
                    errs.push_back(pick(i_ydist, osent[i][t + 1]));
                }
            }
        }

        Expression i_nerr = sum(errs);

        turnid++;
        return -i_nerr;
    };

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
                i_x_x = input(cg, { (long)hidden_dim[DECODER_LAYER] }, &zero);
            v_x_t.push_back(i_x_x);
        }
        i_x_t = concatenate_cols(v_x_t);

        Expression i_y_t = decoder.add_input(i_x_t);

        return i_y_t;
    };
};


}; // namespace cnn
