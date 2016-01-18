#include "cnn/dnn.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "cnn/nodes.h"
#include "cnn/expr-xtra.h"

using namespace std;
using namespace cnn::expr;
using namespace cnn;

namespace cnn {

    enum { X2C = 0, X2CB, X2CI, X2CIB };

    void ClsBasedBuilder::display(ComputationGraph& cg) {
        for (unsigned i = 0; i < layers; ++i) {
            const std::vector<Expression>& vars = param_vars[i];
            for (size_t i = 0; i < vars.size(); i++)
                display_value(vars[i], cg);
        }
    }

    ClsBasedBuilder::ClsBasedBuilder(const vector<int>& cls2nbrwords, /// #words for each class, class starts from 0
        const vector<long> & acc_cls2size, /// the accumulated class size
        const vector<long>& word2cls,
        const vector<long>& dict_wrd_id2within_class_id,
        Model& model,
        cnn::real iscale,
        string name) : 
        clssize(cls2nbrwords), word2cls(word2cls), acc_cls2size(acc_cls2size), dict_wrd_id2within_class_id(dict_wrd_id2within_class_id)
    {
        unsigned int n_cls = clssize.size();
        p_cls = model.add_parameters({ n_cls, HIDDEN_DIM }, iscale, name + " to cls");
        p_cls_bias = model.add_parameters({ n_cls }, iscale, name + " to cls bias");
        for (size_t id = 0; id < n_cls; id++)
        {
            unsigned int  clssize = cls2nbrwords[id];
            p_R.push_back(model.add_parameters({ clssize, HIDDEN_DIM }, iscale, name + " to cls " + id));
            p_bias.push_back(model.add_parameters({ clssize }, iscale, name + " to cls bias " + id));
        }
    }

    void ClsBasedBuilder::new_graph_impl(ComputationGraph& cg) {
        param_vars.clear();
        i_cls = parameter(cg, p_cls);
        i_cls_bias = parameter(cg, p_cls_bias);
        for (unsigned i = 0; i < clssize.size(); ++i) {
            Parameters* p_x2h = p_R[i];
            Parameters* p_x2hb = p_bias[i];
            Expression i_x2h = parameter(cg, p_x2h);
            Expression i_x2hb = parameter(cg, p_x2hb);
            vector<Expression> vars = { i_x2h, i_x2hb };

            param_vars.push_back(vars);
        }
        set_data_in_parallel(1);
    }

    void DNNBuilder::set_data_in_parallel(int n)
    {
        dparallel = n;

        biases.clear();
        for (unsigned i = 0; i < layers; ++i) {
            const vector<Expression>& vars = param_vars[i];
            Expression bimb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[X2HB]));

            vector<Expression> b = { bimb };
            biases.push_back(b);
        }
    }

    Expression DNNBuilder::add_input_impl(const Expression &in) {
        h.resize(layers);

        Expression x = in;

        for (unsigned i = 0; i < layers; ++i) {
            const vector<Expression>& vars = param_vars[i];

            Expression y = affine_transform({ biases[i][0], vars[0], x });

            x = h[i] = tanh(y);
        }
        return h.back();
    }


    void DNNBuilder::copy(const DNNBuilder & rnn) {
        const DNNBuilder& rnn_simple = (const DNNBuilder&)rnn;
        assert(params.size() == rnn_simple.params.size());
        for (size_t i = 0; i < rnn_simple.params.size(); ++i) {
            params[i][0]->copy(*rnn_simple.params[i][0]);
            params[i][1]->copy(*rnn_simple.params[i][1]);
        }
    }

    Expression ReluDNNBuilder::add_input_impl(const Expression &in)  {
        h.resize(layers);

        Expression x = in;

        for (unsigned i = 0; i < layers; ++i) {
            const vector<Expression>& vars = param_vars[i];

            Expression y = affine_transform({ biases[i][0], vars[0], x });

            x = h[i] = rectify(y);
        }
        return h.back();
    }


} // namespace cnn
