#ifndef ENC_DEC_H_
#define ENC_DEC_H_

#include "cnn/cnn.h"
#include "cnn/rnn-state-machine.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/gru.h"
#include <algorithm>
#include <random>
#include <stack>
#include "cnn/data-util.h"
#include <boost/random/random_device.hpp>
#include <boost/random/normal_distribution.hpp>

using namespace cnn::expr;
using namespace std;

namespace cnn {

class Model;

#define ENCODER_LAYER 0
#define INTENTION_LAYER 1
#define DECODER_LAYER 2  
#define ALIGN_LAYER 3

enum { X2Mean = 0, X2MeanBias, X2LogVar, X2LogVarBias };

// interface for constructing an encoder
template<class Builder>
class EncModel{
protected:
    LookupParameters* p_cs;
    Parameters* p_bias;
    Parameters* p_R;  // for affine transformation after decoder

    vector<Parameters*> p_parameters;
    vector<Expression> v_parameters_exp;

    /// context dimension to decoder dimension
    Parameters* p_cxt2dec_w;
    Expression i_cxt2dec_w;

    int decoder_use_additional_input;

    size_t layers;
    Builder encoder_fwd, encoder_bwd; /// for encoder at each turn

    /// for different slices
    vector<Builder*> v_encoder_fwd, v_encoder_bwd;

    /// for alignment to source
    Parameters* p_U;
    Expression i_U;

    Model model;

    int vocab_size;
    vector<unsigned> hidden_dim;
    int rep_hidden;

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

    size_t nutt; // for multiple training utterance per inibatch
    vector<cnn::real> zero;
public:
    /// for criterion
    vector<Expression> v_errs;
    size_t src_words;
    size_t tgt_words;

public:
    EncModel() {};
    EncModel(cnn::Model& model, int layers, int vocab_size_src, const vector<unsigned>& hidden_dims, int hidden_replicates, int decoder_use_additional_input = 0, int mem_slots = 0, float iscale = 1.0) :
        layers(layers),
        encoder_fwd(layers, hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], &model, iscale),
        encoder_bwd(layers, hidden_dims[ENCODER_LAYER], hidden_dims[ENCODER_LAYER], &model, iscale),
        decoder_use_additional_input(decoder_use_additional_input),
        vocab_size(vocab_size_src),
        rep_hidden(hidden_replicates)
    {
        hidden_dim = hidden_dims;

        if (hidden_dim[DECODER_LAYER] != hidden_dim[ENCODER_LAYER])
        {
            cerr << "wrong dimension of encoder and decoder layer. they should be the same, as they use the same lookup table" << endl;
            throw("wrong dimension of encoder and decoder layer. they should be the same, as they use the same lookup table");
        }

        p_cs = model.add_lookup_parameters(long(vocab_size_src), { long(hidden_dim[ENCODER_LAYER]) }, iscale);

        p_parameters = {};
        p_R = model.add_parameters({ long(vocab_size_src), long(hidden_dim[DECODER_LAYER]) }, iscale);
        p_bias = model.add_parameters({ long(vocab_size_src) }, iscale);

        /// parameter for prediction mean/variance
        p_parameters.push_back(model.add_parameters({ long(hidden_dim[DECODER_LAYER]), long(hidden_dim[DECODER_LAYER]) * 2 }, iscale));
        p_parameters.push_back(model.add_parameters({ long(hidden_dim[DECODER_LAYER]) }, iscale));
        p_parameters.push_back(model.add_parameters({ long(hidden_dim[DECODER_LAYER]), long(hidden_dim[DECODER_LAYER]) * 2 }, iscale));
        p_parameters.push_back(model.add_parameters({ long(hidden_dim[DECODER_LAYER]) }, iscale));

        p_U = model.add_parameters({ long(hidden_dim[ALIGN_LAYER]), long(2 * hidden_dim[ENCODER_LAYER]) }, iscale);

        for (size_t i = 0; i < layers * rep_hidden; i++)
        {
            p_h0.push_back(model.add_parameters({ long(hidden_dim[ENCODER_LAYER]) }, iscale));
            p_h0.back()->reset_to_zero();
        }
        zero.resize(hidden_dim[ENCODER_LAYER], 0);  /// for the no obs observation

        p_cxt2dec_w = model.add_parameters({ long(hidden_dim[DECODER_LAYER]), long(hidden_dim[INTENTION_LAYER]) }, iscale);

        i_h0.clear();
    };

    ~EncModel(){};

    /// for context
    void reset()
    {
        turnid = 0;

        v_encoder_bwd.clear();
        v_encoder_fwd.clear();

        i_h0.clear();

        v_errs.clear();
        src_words = 0;
        tgt_words = 0;

        v_parameters_exp.clear();
    }

    virtual void assign_cxt(ComputationGraph &cg, size_t nutt)
    {
        if (turnid <= 0)
        {
            /// no information from previous turns
            reset();
            return;
        }

        /// prepare for the next run
        v_encoder_bwd.clear();
        v_encoder_fwd.clear();

        i_h0.clear();
        v_errs.clear();
        tgt_words = 0;
        src_words = 0;

    }

    virtual void assign_cxt(ComputationGraph &cg, size_t nutt,
        vector<vector<cnn::real>>& v_last_cxt_s,
        vector<vector<cnn::real>>& v_last_decoder_s)
    {
        if (turnid <= 0)
        {
            /// no information from previous turns
            reset();
            return;
        }

        /// prepare for the next run
        v_encoder_bwd.clear();
        v_encoder_fwd.clear();

        i_h0.clear();
        v_errs.clear();
        tgt_words = 0;
        src_words = 0;

    }

public:
    /// run in batch with multiple sentences
    /// source [utt][data stream] is utterance first and then its content
    /// the context RNN uses the last state of the encoder RNN as its input
    virtual Expression start_new_instance(const std::vector<std::vector<int>> &source, ComputationGraph &cg)
    {
        nutt = source.size();

        std::vector<Expression> source_embeddings;

        v_encoder_fwd.push_back(new Builder(encoder_fwd));
        v_encoder_bwd.push_back(new Builder(encoder_bwd));

        v_encoder_fwd.back()->new_graph(cg);
        v_encoder_fwd.back()->set_data_in_parallel(nutt);
        v_encoder_fwd.back()->start_new_sequence();

        v_encoder_bwd.back()->new_graph(cg);
        v_encoder_bwd.back()->set_data_in_parallel(nutt);
        v_encoder_bwd.back()->start_new_sequence();

        /// the source sentence has to be approximately the same length
        src_len = each_sentence_length(source);
        src_fwd = bidirectional<Builder>(slen, source, cg, p_cs, zero, v_encoder_fwd.back(), v_encoder_bwd.back(), hidden_dim[ENCODER_LAYER]);

        v_src = shuffle_data(src_fwd, (size_t)nutt, (size_t)2 * hidden_dim[ENCODER_LAYER], src_len);

        /// for contet
        vector<Expression> to;
        /// take the top layer from decoder, take its final h
        to.push_back(v_encoder_fwd.back()->final_h()[layers - 1]);
        to.push_back(v_encoder_bwd.back()->final_h()[layers - 1]);

        Expression q_m = concatenate(to);

        return concatenate(to); 
    };

    std::vector<Expression> build_graph(const std::vector<std::vector<int>> &source, ComputationGraph &cg)
    {
        size_t nutt;
        vector<Expression> outputs;

        if (v_parameters_exp.size() == 0){
            for (auto &p : p_parameters)
            {
                v_parameters_exp.push_back(parameter(cg, p));
            }
        }

        Expression encoded_source = start_new_instance(source, cg);

        Expression mu = v_parameters_exp[X2Mean] * encoded_source + v_parameters_exp[X2MeanBias];
        Expression std = exp(0.5 * (v_parameters_exp[X2LogVar] * encoded_source + v_parameters_exp[X2LogVarBias]));

        outputs.push_back(mu);
        outputs.push_back(std);

        turnid++;
        return outputs;
    };

    std::vector<Expression> build_graph(const std::vector<std::vector<int>> &source1, const std::vector<std::vector<int>> &source2, ComputationGraph &cg){
        /// assume that the previous input has no effects on this 
        /// independent encoding assumption
        return build_graph(source2, cg);
    };

    /**
    generate data using mean and variance 
    */
    vector<vector<cnn::real>> sample(vector<Expression> & mean_var, ComputationGraph& cg, size_t nsamples = 1)
    {
        assert(mean_var.size() == 2); /// only generates from Gaussian distribution [mu, log(sigma)]
        vector<vector<cnn::real>> samples;
        vector<cnn::real> vmean = get_value(mean_var[0], cg);
        vector<cnn::real> vstd  = get_value(mean_var[1], cg);
        size_t dim = vmean.size();
        vector<cnn::real> vec;

        boost::random::random_device rng;
        boost::random::normal_distribution<cnn::real> generator(0, 1.0);

        for (size_t k = 0; k < nsamples; k++)
        {
            /// generate random sample given mean and variance
            vec.resize(dim);
            for (size_t l = 0; l < dim; l ++)
            {
                vec[l] = generator(rng) * vstd[l];
                vec[l] += vmean[l];
            }

            samples.push_back(vec);
        }

        return samples;
    }
};

} // namespace cnn

#endif
