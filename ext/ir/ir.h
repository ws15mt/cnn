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
#include "ext/dialogue/dialogue.h"
#include "cnn/approximator.h"
#include "ext/lda/lda.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

namespace cnn {

    /** End-to-End modeling 
    For each state, output a softmax of all possible candidate responses
    State to state transition is a Markov model, or RNN. 
    Different from AWI model in the sense of using sentence-level responses, whereas in AWI, the response is generation. 
    */
    template <class Builder, class Decoder>
    class ClassificationEncoderDecoder: public DialogueBuilder<Builder, Decoder>{
    protected:
        int cls_size; /// number of classes for output
        int vocab_size; /// input vocabulary size

        Parameters * p_cxt_to_decoder, *p_enc_to_intention;
        Expression i_cxt_to_decoder, i_enc_to_intention;

        Builder combiner; /// the combiner that combines the multipe sources of inputs, and possibly its history

    public:
        ClassificationEncoderDecoder(Model& model,
            unsigned vocab_size_src, unsigned vocab_size_tgt, const vector<unsigned int>& layers,
            const vector<unsigned>& hidden_dims, unsigned hidden_replicates, unsigned additional_input = 0, unsigned mem_slots = 0, cnn::real iscale = 1.0) :
            DialogueBuilder(model, vocab_size_src, vocab_size_tgt, layers, hidden_dims, hidden_replicates, additional_input, mem_slots, iscale),
            combiner(layers[INTENTION_LAYER], vector<unsigned>{hidden_dims[INTENTION_LAYER], hidden_dims[INTENTION_LAYER]}, &model, iscale),
            cls_size(vocab_size_tgt),
            vocab_size(vocab_size_src)
        {
            p_cxt_to_decoder = model.add_parameters({ hidden_dim[DECODER_LAYER], hidden_dim[INTENTION_LAYER] }, iscale, "p_cxt_to_decoder");

            p_enc_to_intention = model.add_parameters({ hidden_dim[INTENTION_LAYER], hidden_dim[ENCODER_LAYER] }, iscale, "p_enc_to_intention");
        }

        ~ClassificationEncoderDecoder()
        {
        }

        void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg)
        {
            nutt = source.size();
            std::vector<Expression> v_tgt2enc;

            i_h0.resize(p_h0.size());

            for (int k = 0; k < p_h0.size(); k++)
            {
                i_h0[k] = concatenate_cols(vector<Expression>(nutt, parameter(cg, p_h0[k])));
            }

            combiner.new_graph(cg);
            combiner.start_new_sequence();
            combiner.set_data_in_parallel(nutt);

            i_R = parameter(cg, p_R); // hidden -> word rep parameter
            i_bias = parameter(cg, p_bias);

            i_cxt_to_decoder = parameter(cg, p_cxt_to_decoder);
            i_enc_to_intention = parameter(cg, p_enc_to_intention);

            /// take the previous response as input
            encoder_fwd.new_graph(cg);
            encoder_fwd.set_data_in_parallel(nutt);
            encoder_fwd.start_new_sequence(i_h0);

            /// encode the source side input, with intial state from the previous response
            /// this is a way to combine the previous response and the current input
            /// notice that for just one run of the RNN, 
            /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
            encoder_bwd.new_graph(cg);
            encoder_bwd.set_data_in_parallel(nutt);
            encoder_bwd.start_new_sequence();

            /// get the raw encodeing from source
            src_fwd = concatenate_cols(average_embedding(slen, source, cg, p_cs));

            encoder_bwd.add_input(src_fwd);

            /// update intention, with inputs for each layer for combination
            /// the context hidden state is tanh(W_h previous_hidden + encoder_bwd.final_s at each layer)
            vector<Expression> v_to_intention;
            for (auto p : encoder_bwd.final_s())
                v_to_intention.push_back(i_enc_to_intention* p);
            combiner.add_input(v_to_intention);

            /// decoder start with a context from intention 
            decoder.new_graph(cg);
            decoder.set_data_in_parallel(nutt);
            decoder.add_input(i_cxt_to_decoder * combiner.back());  /// get the intention
        }
        
        void start_new_instance(const std::vector<std::vector<int>> &prv_response,
            const std::vector<std::vector<int>> &source,
            ComputationGraph &cg)
        {
            nutt = source.size();
            std::vector<Expression> v_tgt2enc;

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
            }

            /// take the previous response as input
            encoder_fwd.new_graph(cg);
            encoder_fwd.set_data_in_parallel(nutt);
            encoder_fwd.start_new_sequence(i_h0);

            /// encode the source side input, with intial state from the previous response
            /// this is a way to combine the previous response and the current input
            /// notice that for just one run of the RNN, 
            /// the state is changed to tanh(W_prev prev_response + W_input current_input) for each layer
            encoder_bwd.new_graph(cg);
            encoder_bwd.set_data_in_parallel(nutt);
            encoder_bwd.start_new_sequence();

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
            decoder.add_input(i_cxt_to_decoder * combiner.back());  /// get the intention
        }

        Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg)
        {
            return decoder.back(); 
        }

        vector<Expression> build_graph(
            const std::vector<std::vector<int>> &current_user_input,
            const std::vector<std::vector<int>>& target_response,
            ComputationGraph &cg)
        {
            for (auto& p : target_response){
                if (p.size() != 1)
                    throw("classification task, shouldn't be a sequence output");
            }

            unsigned int nutt = current_user_input.size();
            start_new_instance(std::vector<std::vector<int>>(), current_user_input, cg);

            vector<vector<Expression>> this_errs(nutt);
            vector<Expression> errs;

            Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
            Expression i_bias = parameter(cg, p_bias);  // word bias

            v_decoder_context.clear();
            v_decoder_context.resize(nutt);

            Expression i_h_t = decoder_step(target_response[0], cg); /// get the top layer output

            Expression i_r_t = i_R * i_h_t;
            Expression i_ydist = log_softmax(i_r_t);

            Expression i_reshaped_t = reshape(i_ydist, { cls_size * nutt });
            for (int i = 0; i < nutt; i++)
            {
//                Expression r_r_t = pickrange(x_r_t, i * cls_size, (i + 1)*cls_size);
//                Expression i_ydist = log_softmax(r_r_t);
                this_errs[i].push_back(-pick(i_reshaped_t, target_response[i][0] + cls_size * i));
                tgt_words++;
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
            for (auto& p : target_response){
                if (p.size() != 1)
                    throw("classification task, shouldn't be a sequence output");
            }

            unsigned int nutt = current_user_input.size();
            start_new_instance(prv_response, current_user_input, cg);

            vector<vector<Expression>> this_errs(nutt);
            vector<Expression> errs;

            Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
            Expression i_bias = parameter(cg, p_bias);  // word bias

            v_decoder_context.clear();
            v_decoder_context.resize(nutt);

            Expression i_h_t = decoder_step(target_response[0], cg); /// get the top layer output

            Expression i_r_t = i_R * i_h_t;
            Expression x_r_t = reshape(i_r_t, { cls_size * nutt });

            for (int i = 0; i < nutt; i++)
            {
                Expression r_r_t = pickrange(x_r_t, i * cls_size, (i + 1)*cls_size);
                Expression i_ydist = log_softmax(r_r_t);
                this_errs[i].push_back(-pick(i_ydist, target_response[i][0]));
                tgt_words++;
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

        void assign_cxt(ComputationGraph &cg, unsigned int nutt)
        {
            if (turnid <= 0 || last_cxt_s.size() == 0)
            {
                /// no information from previous turns
                reset();
                return;
            }

            last_context_exp.clear();
            for (const auto &p : last_cxt_s)
            {
                Expression iv;
                if (nutt > 1)
                    iv = input(cg, { (unsigned int)p.size() / nutt, nutt }, &p);
                else
                    iv = input(cg, { (unsigned int)p.size() }, &p);
                last_context_exp.push_back(iv);
            }
            /// prepare for the next run
            i_h0.clear();
            v_errs.clear();
            tgt_words = 0;
            src_words = 0;
        }

        void assign_cxt(ComputationGraph &cg, unsigned int nutt,
            vector<vector<cnn::real>>& v_last_s, vector<vector<cnn::real>>& v_decoder_s){
            throw("not implemented"); 
        }

        void serialise_context(ComputationGraph& cg,
            vector<vector<cnn::real>>& v_last_cxt_s,
            vector<vector<cnn::real>>& v_last_decoder_s)
        {
            /// get the top output
            vector<vector<cnn::real>> vm;

            vm.clear();
            for (const auto &p : combiner.final_s())
            {
                vm.push_back(get_value(p, cg));
            }
            last_cxt_s = vm;
            v_last_cxt_s = last_cxt_s;
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
            vector<vector<cnn::real>> vm;

            vm.clear();
            for (const auto &p : combiner.final_s())
            {
                vm.push_back(get_value(p, cg));
            }
            last_cxt_s = vm;
        }

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
            int t = 0;
            Sentence prv_response;

            start_new_single_instance(prv_response, source, cg);

            Expression i_bias = parameter(cg, p_bias);
            Expression i_R = parameter(cg, p_R);

            v_decoder_context.clear();

            Expression i_y_t = decoder_single_instance_step(0, cg);
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

            target.push_back(w);
            
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
            int t = 0;

            start_new_single_instance(prv_response, source, cg);

            Expression i_bias = parameter(cg, p_bias);
            Expression i_R = parameter(cg, p_R);

            v_decoder_context.clear();

            Expression i_y_t = decoder_single_instance_step(0, cg);
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

            target.push_back(w);
            
            v_decoder_context.push_back(decoder.final_s());
            save_context(cg);
            serialise_context(cg);

            turnid++;
            return target;
        }
    };




}; // namespace cnn
