#ifndef DIALOGUE_
#define DIALOGUE_

#include "cnn/cnn.h"
#include "cnn/rnn-state-machine.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/gru.h"
#include <algorithm>
#include <stack>
#include "cnn/data-util.h"

#define UNDERSTAND_AWI
#define UNDERSTAND_AWI_ADD_ATTENTION

using namespace cnn::expr;
using namespace std;

namespace cnn {

class Model;

#define ENCODER_LAYER 0
#define INTENTION_LAYER 1
#define DECODER_LAYER 2  
#define ALIGN_LAYER 3

// interface for constructing an a dialogue 
template<class Builder, class Decoder>
class DialogueBuilder{
protected:
    LookupParameters* p_cs;
    Parameters* p_bias;
    Parameters* p_R;  // for affine transformation after decoder

    /// context dimension to decoder dimension
    Parameters* p_cxt2dec_w;
    Expression i_cxt2dec_w;

    vector<size_t> layers;
    Decoder decoder;  // for decoder at each turn
    Builder encoder_fwd, encoder_bwd; /// for encoder at each turn
    Builder context; // for contexter

    /// for different slices
    vector<Builder*> v_encoder_fwd, v_encoder_bwd;
    vector<Decoder*> v_decoder;

    /// for alignment to source
    Parameters* p_U;
    Expression i_U;

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

    /// from previous target to context
    vector<vector<Expression>> v_decoder_context; //// [nutt][rep_hidden * layers] 
    vector<Expression> to_cxt;  /// this is the final_s from decoder RNNm

    map<int, vector<vector<cnn::real>>> to_cxt_value; /// memory of to_cxt
    vector<vector<cnn::real>> last_cxt_s;  /// memory of context history for LSTM including h and c, use this for initialization of intent RNN
    vector<vector<cnn::real>> last_decoder_s;  /// memory of target side decoder history for LSTM including h and c, use this for initialization of source side RNN
    vector<Expression> last_context_exp;  /// expression to the last context hidden state

    size_t turnid;

    size_t nutt; // for multiple training utterance per inibatch
    vector<cnn::real> zero;
public:
    /// for criterion
    vector<Expression> v_errs;
    size_t src_words;
    size_t tgt_words;

public:
    DialogueBuilder() {};
    DialogueBuilder(cnn::Model& model, int vocab_size_src, int vocab_size_tgt, const vector<size_t>& layers, const vector<unsigned>& hidden_dims, int hidden_replicates, int decoder_use_additional_input = 0, int mem_slots = 0, cnn::real iscale = 1.0) :
        layers(layers),
        decoder(layers[DECODER_LAYER], hidden_dims[DECODER_LAYER] + decoder_use_additional_input * hidden_dims[ENCODER_LAYER], hidden_dims[DECODER_LAYER], &model, iscale),
        encoder_fwd(layers[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], &model, iscale),
        encoder_bwd(layers[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], &model, iscale),
        decoder_use_additional_input(decoder_use_additional_input),
        context(layers[INTENTION_LAYER], layers[ENCODER_LAYER] * hidden_replicates * hidden_dims[ENCODER_LAYER], hidden_dims[INTENTION_LAYER], &model, iscale),
        vocab_size(vocab_size_src), vocab_size_tgt(vocab_size_tgt),
        rep_hidden(hidden_replicates)
    {
        hidden_dim = hidden_dims;

        if (hidden_dim[DECODER_LAYER] != hidden_dim[ENCODER_LAYER])
        {
            cerr << "wrong dimension of encoder and decoder layer. they should be the same, as they use the same lookup table" << endl;
            throw("wrong dimension of encoder and decoder layer. they should be the same, as they use the same lookup table");
        }

        p_cs = model.add_lookup_parameters(long(vocab_size_src), { long(hidden_dim[ENCODER_LAYER]) }, iscale);
        p_R = model.add_parameters({ long(vocab_size_tgt), long(hidden_dim[DECODER_LAYER]) }, iscale);
        p_bias = model.add_parameters({ long(vocab_size_tgt) }, iscale);

        p_U = model.add_parameters({ long(hidden_dim[ALIGN_LAYER]), long(2 * hidden_dim[ENCODER_LAYER]) }, iscale);

        for (size_t i = 0; i < layers[ENCODER_LAYER] * rep_hidden; i++)
        {
            p_h0.push_back(model.add_parameters({ long(hidden_dim[ENCODER_LAYER]) }, iscale));
            p_h0.back()->reset_to_zero();
        }
        zero.resize(hidden_dim[ENCODER_LAYER], 0);  /// for the no obs observation

        p_cxt2dec_w = model.add_parameters({ long(hidden_dim[DECODER_LAYER]), long(hidden_dim[INTENTION_LAYER]) }, iscale);

        i_h0.clear();
    };

    ~DialogueBuilder(){};

    /// for context
    void reset()
    {
        last_cxt_s.clear();
        turnid = 0;

        to_cxt.clear();
        to_cxt_value.clear();

        last_context_exp.clear();

        v_encoder_bwd.clear();
        v_encoder_fwd.clear();
        v_decoder.clear();

        i_h0.clear();

        v_errs.clear();
        src_words = 0;
        tgt_words = 0;
    }

    virtual void serialise_context(ComputationGraph& cg,
        vector<vector<cnn::real>>& v_last_cxt_s,
        vector<vector<cnn::real>>& v_last_decoder_s)
    {
        /// get the top output
        vector<vector<cnn::real>> vm;

        vm.clear();
        for (const auto &p : context.final_s())
        {
            vm.push_back(get_value(p, cg));
        }
        last_cxt_s = vm;
        v_last_cxt_s = last_cxt_s;

        vector<vector<cnn::real>> v_last_d;
        size_t nutt = v_decoder_context.size();
        size_t ndim = v_decoder_context[0].size();
        v_last_d.resize(ndim);

        size_t ik = 0;
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
        v_last_decoder_s = last_decoder_s;
    }

    virtual void serialise_context(ComputationGraph& cg)
    {
        /// get the top output
        vector<vector<cnn::real>> vm;

        vm.clear();
        for (const auto &p : context.final_s())
        {
            vm.push_back(get_value(p, cg));
        }
        last_cxt_s = vm;

        vector<vector<cnn::real>> v_last_d;
        size_t nutt = v_decoder_context.size();
        size_t ndim = v_decoder_context[0].size();
        v_last_d.resize(ndim);

        size_t ik = 0;
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

    virtual void save_context(ComputationGraph& cg)
    {
        to_cxt.clear();
        vector<Expression> ve;
        vector<vector<Expression>> vve;
        size_t ndim = v_decoder_context[0].size();
        vve.resize(ndim);
        for (const auto&p : v_decoder_context)
        {
            /// for each utt
            size_t ik = 0;
            for (auto pt : p)
            {
                vve[ik].push_back(pt);
                ik++;
            }
        }
        for (const auto &p : vve)
            to_cxt.push_back(concatenate_cols(p));
    }

    virtual void assign_cxt(ComputationGraph &cg, size_t nutt)
    {
        if (turnid <= 0 || last_cxt_s.size() == 0 || last_decoder_s.size() == 0)
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
                iv = input(cg, { long(p.size() / nutt), long(nutt) }, &p);
            else
                iv = input(cg, { long(p.size()) }, &p);
            last_context_exp.push_back(iv);
        }

        vector<Expression> v_last_d;
        for (const auto &p : last_decoder_s)
        {
            Expression iv;
            if (nutt > 1)
                iv = input(cg, { long(p.size() / nutt), long(nutt) }, &p);
            else
                iv = input(cg, { long(p.size()) }, &p);
            v_last_d.push_back(iv);
        }

        to_cxt = v_last_d;

        /// prepare for the next run
        v_encoder_bwd.clear();
        v_encoder_fwd.clear();
        v_decoder.clear();
        i_h0.clear();
        v_errs.clear();
        tgt_words = 0;
        src_words = 0;

    }

    /**
    assign observation to the hidden latent varaible

    @v_last_cxt_s [parameter_index][values vector for this parameter]. this is to the context or intention network
    @v_last_decoder_s [parameter_index][values vector for this parameter]. this is to the decoder network
    */
    virtual void assign_cxt(ComputationGraph &cg, size_t nutt,
        vector<vector<cnn::real>>& v_last_cxt_s, 
        vector<vector<cnn::real>>& v_last_decoder_s)
    {
        if (turnid <= 0 || v_last_cxt_s.size() == 0 || v_last_decoder_s.size() == 0)
        {
            /// no information from previous turns
            reset();
            return;
        }

        last_context_exp.clear();
        for (const auto &p : v_last_cxt_s)
        {
            Expression iv;
            if (nutt > 1)
                iv = input(cg, { long(p.size() / nutt), long(nutt) }, &p);
            else
                iv = input(cg, { long(p.size()) }, &p);
            last_context_exp.push_back(iv);
        }

        vector<Expression> v_last_d;
        for (const auto &p : v_last_decoder_s)
        {
            Expression iv;
            if (nutt > 1)
                iv = input(cg, { long(p.size() / nutt), long(nutt) }, &p);
            else
                iv = input(cg, { long(p.size()) }, &p);
            v_last_d.push_back(iv);
        }

        to_cxt = v_last_d;

        /// prepare for the next run
        v_encoder_bwd.clear();
        v_encoder_fwd.clear();
        v_decoder.clear();
        i_h0.clear();
        v_errs.clear();
        tgt_words = 0;
        src_words = 0;

    }

    virtual void assign_cxt(ComputationGraph &cg, size_t nutt,
        std::vector<std::vector<std::vector<cnn::real>>>& v_last_cxt_s)
    {
    }

    virtual void assign_cxt(ComputationGraph &cg,
        const vector<vector<int>>& v_last_cxt_s)
    {}

    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
    {
        const int sos_sym = tdict.Convert("<s>");
        const int eos_sym = tdict.Convert("</s>");

        std::vector<int> target;
        target.push_back(sos_sym);

        //  std::cerr << tdict.Convert(target.back());
        int t = 0;

        start_new_instance(source, cg);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();
        while (target.back() != eos_sym)
        {
            Expression i_y_t = decoder_step(target.back(), cg);
            Expression i_r_t = i_bias + i_R * i_y_t;
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

        vector<Expression> v_t = decoder.final_s();

        save_context(cg);

        turnid++;

        return target;
    }

#ifdef INPUT_UTF8
    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict<std::wstring> &tdict,
        Builder* encoder_fwd, Builder* encoder_bwd,
        Builder * context,
        Builder *decoder)
#else
    std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict,
        Builder* encoder_fwd, Builder* encoder_bwd,
        Builder * context,
        Builder *decoder)
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

        //  std::cerr << tdict.Convert(target.back());
        int t = 0;

        start_new_instance(source, cg, encoder_fwd, encoder_bwd, context, decoder);

        Expression i_bias = parameter(cg, p_bias);
        Expression i_R = parameter(cg, p_R);

        v_decoder_context.clear();
        v_decoder_context.resize(1);
        while (target.back() != eos_sym)
        {
            vector<int> vt(1, target.back());
            Expression i_y_t = decoder_step(vt, cg, decoder);
            Expression i_r_t = i_bias + i_R * i_y_t;
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

        vector<Expression> v_t = decoder->final_s();
        v_decoder_context[0] = v_t;

        save_context(cg);

        turnid++;

        return target;
    }

    std::vector<int> decode_tuple(const SentenceTuple&source, ComputationGraph& cg, cnn::Dict  &sdict, cnn::Dict  &tdict)
    {
        vector<int> vres;
        throw("not implemented");
        return vres;
    };

 protected:
     /// run in batch with multiple sentences
     /// source [utt][data stream] is utterance first and then its content
     /// the context RNN uses the last state of the encoder RNN as its input
     virtual void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg) {
         nutt = source.size();

         if (i_h0.size() == 0)
         {
             i_h0.clear();
             for (auto p : p_h0)
             {
                 i_h0.push_back(concatenate_cols(vector<Expression>(nutt,parameter(cg, p))));
             }
             context.new_graph(cg);
             context.start_new_sequence();

             i_cxt2dec_w = parameter(cg, p_cxt2dec_w);
         }

         size_t n_turns = 0;
         std::vector<Expression> source_embeddings;

         context.set_data_in_parallel(nutt);

         v_encoder_fwd.push_back(new Builder(encoder_fwd));
         v_encoder_bwd.push_back(new Builder(encoder_bwd));
         v_decoder.push_back(new Decoder(decoder));

         n_turns = v_encoder_fwd.size(); 

         v_encoder_fwd.back()->new_graph(cg);
         v_encoder_fwd.back()->set_data_in_parallel(nutt);

         if (n_turns > 1)
             v_encoder_fwd.back()->start_new_sequence(context.final_s());
         else
             v_encoder_fwd.back()->start_new_sequence(i_h0);

         v_encoder_bwd.back()->new_graph(cg);
         v_encoder_bwd.back()->set_data_in_parallel(nutt);
         if (n_turns > 1)
             v_encoder_bwd.back()->start_new_sequence(context.final_s());
         else
             v_encoder_bwd.back()->start_new_sequence(i_h0);

         /// the source sentence has to be approximately the same length
         src_len = each_sentence_length(source);

         src_fwd = bidirectional<Builder>(slen, source, cg, p_cs, zero, *v_encoder_fwd.back(), *v_encoder_bwd.back(), hidden_dim[ENCODER_LAYER]);

         v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)2 * hidden_dim[ENCODER_LAYER], src_len);

         i_U = parameter(cg, p_U);
         src = i_U * concatenate_cols(v_src);  // precompute 

         v_decoder.back()->new_graph(cg);
         v_decoder.back()->set_data_in_parallel(nutt);
         if (n_turns > 1)
             v_decoder.back()->start_new_sequence(context.final_s());  /// get the intention
         else
             v_decoder.back()->start_new_sequence(i_h0);
     };

     /// run in batch with multiple sentences
     /// source [utt][data stream] is utterance first and then its content
     /// the context RNN uses the last state of the encoder RNN as its input
     virtual void start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg,
         Builder* encoder_fwd, Builder* encoder_bwd, 
         Builder * context, 
         Decoder *decoder) 
     {
         nutt = source.size();

         std::vector<Expression> source_embeddings;

         context->set_data_in_parallel(nutt);

         encoder_fwd->new_graph(cg);
         encoder_fwd->set_data_in_parallel(nutt);
#ifndef UNDERSTAND_AWI
         if (to_cxt.size() > 0)
             encoder_fwd->start_new_sequence(to_cxt);
         else
#endif
             encoder_fwd->start_new_sequence();

         encoder_bwd->new_graph(cg);
         encoder_bwd->set_data_in_parallel(nutt);
#ifndef UNDERSTAND_AWI
         if (to_cxt.size() > 0)
             encoder_bwd->start_new_sequence(to_cxt);
         else
#endif
             encoder_bwd->start_new_sequence();

         /// the source sentence has to be approximately the same length
         src_len = each_sentence_length(source);
         //         if (!similar_length(source))
         //         {
         //             cerr << "sentence length differs too much" << endl;
         //             abort();
         //         }
         src_fwd = bidirectional<Builder>(slen, source, cg, p_cs, zero, *encoder_fwd, *encoder_bwd, hidden_dim[ENCODER_LAYER]);

         v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)2 * hidden_dim[ENCODER_LAYER], src_len);

         /// for contet
         vector<Expression> to;
         /// take the top layer from decoder, take its final h
         to.push_back(encoder_fwd->final_h()[layers[ENCODER_LAYER] - 1]);
         to.push_back(encoder_bwd->final_h()[layers[ENCODER_LAYER] - 1]);

         Expression q_m = concatenate(to);

#ifndef UNDERSTAND_AWI
         if (to_cxt.size() > 0)
         {
             Expression i_tgt2cxt = parameter(cg, p_tgt2cxt);
             context->add_input(q_m + i_tgt2cxt * to_cxt.back());
         }
         else
#endif
             context->add_input(q_m);
         cg.incremental_forward();

         vector<Expression> d_m = context->final_s();

         i_U = parameter(cg, p_U);
         src = i_U * concatenate_cols(v_src);  // precompute 

         decoder->new_graph(cg);
         decoder->set_data_in_parallel(nutt);
         decoder->start_new_sequence(d_m);  /// get the intention
     };

     virtual void start_new_instance(const std::vector<int> &src, ComputationGraph &cg)
     {
         std::vector<std::vector<int>> source(1, src);
         return start_new_instance(source, cg);
     }

     virtual void start_new_instance(const std::vector<int> &src, ComputationGraph &cg,
         Builder * encoder_fwd, 
         Builder * encoder_bwd,
         Builder * context,
         Decoder* decoder)
     {
         std::vector<std::vector<int>> source(1, src);
         return start_new_instance(source, cg, encoder_fwd, encoder_bwd, context, decoder);
     }

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
                 else if (t == osent[i].size() - 1)
                 {
                     /// get the last hidden state to decode the i-th utterance
                     vector<Expression> v_t; 
                     for (auto p : v_decoder.back()->final_s())
                     {
                         Expression i_tt = reshape(p, { (long)(nutt * hidden_dim[DECODER_LAYER]) });
                         int stt = i * hidden_dim[DECODER_LAYER];
                         int stp = stt + hidden_dim[DECODER_LAYER];
                         Expression i_t = pickrange(i_tt, stt, stp);
                         v_t.push_back(i_t);
                     }
                     v_decoder_context[i] = v_t;
                 }
             }
         }

         /// for context
         vector<Expression> to;
         /// take the top layer from decoder, take its final h
         to.push_back(v_encoder_fwd.back()->final_h()[layers[ENCODER_LAYER] - 1]);
         to.push_back(v_encoder_bwd.back()->final_h()[layers[ENCODER_LAYER] - 1]);

         Expression q_m = concatenate(to);

         if (v_decoder.size() > 0)
         {
             context.add_input(q_m + concatenate(v_decoder.back()->final_h()));
         }
         else
             context.add_input(q_m);
         cg.incremental_forward();

         save_context(cg);

         Expression i_nerr = sum(errs);

         turnid++;
         return -i_nerr;
     };

     Expression build_graph(const std::vector<std::vector<int>> &source,
         const std::vector<std::vector<int>>& osent,
         const std::vector<std::vector<int>> &additional_input,
         ComputationGraph &cg)
     {
     }

     public:
     vector<Expression> build_comp_graph(const std::vector<std::vector<int>> &source,
         const std::vector<std::vector<int>>& osent,
         ComputationGraph &cg)
     {
         vector<Expression> verr;
         throw("not implemented");
         return verr;
     }

     /*
     Expression build_graph_target_source(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg){
         return build_graph(source, osent, cg, &t2s_encoder_fwd, &t2s_encoder_bwd, &context, &t2s_decoder);
     }

     Expression build_graph(const std::vector<std::vector<int>> &source, const std::vector<std::vector<int>>& osent, ComputationGraph &cg,
         Builder* encoder_fwd, Builder* encoder_bwd,
         Builder * context,
         Builder *decoder)
     {
         size_t nutt;
         start_new_instance(source, cg, encoder_fwd, encoder_bwd, context, decoder);

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
             Expression i_y_t = decoder_step(vobs, cg, decoder);
             Expression i_r_t = i_bias_mb + i_R * i_y_t;
             cg.incremental_forward();

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
                 else if (t == osent[i].size() - 1)
                 {
                     /// get the last hidden state to decode the i-th utterance
                     vector<Expression> v_t;
                     for (auto p : decoder->final_s())
                     {
                         Expression i_tt = reshape(p, { (long)nutt * hidden_dim });
                         int stt = i * hidden_dim;
                         int stp = stt + hidden_dim;
                         Expression i_t = pickrange(i_tt, stt, stp);
                         v_t.push_back(i_t);
                     }
                     v_decoder_context[i] = v_t;
                 }
             }
         }

         save_context(cg);

         Expression i_nerr = sum(errs);

         turnid++;
         return -i_nerr;
     };
*/

protected:
//    virtual Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg, Builder * decoder) = 0;
    virtual Expression decoder_step(vector<int> trg_tok, ComputationGraph& cg) = 0;
    virtual Expression decoder_step(int trg_tok, ComputationGraph& cg)
    {
        vector<int> input(1, trg_tok);
        return decoder_step(input, cg);
    }
};

} // namespace cnn

#endif
