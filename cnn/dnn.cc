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

    enum { X2H = 0, X2HB };

    void DNNBuilder::display(ComputationGraph& cg) {

        for (unsigned i = 0; i < layers; ++i) {
            const std::vector<Expression>& vars = param_vars[i];
            for (size_t i = 0; i < vars.size(); i++)
                display_value(vars[i], cg);
        }
    }

    DNNBuilder::DNNBuilder(unsigned ilayers,
        unsigned input_dim,
        unsigned hidden_dim,
        Model* model,
        cnn::real iscale,
        string name) 
    {
        layers = ilayers;
        long layer_input_dim = input_dim;
        input_dims = vector<unsigned>(layers, layer_input_dim);

        for (unsigned i = 0; i < layers; ++i) {
            input_dims[i] = layer_input_dim;

            string i_name = "";
            if (name.size() > 0)
                i_name = name + "p_x2h" + boost::lexical_cast<string>(i);
            Parameters* p_x2h = model->add_parameters({ long(hidden_dim), layer_input_dim }, iscale, i_name);
            if (name.size() > 0)
                i_name = name + "p_x2hb" + boost::lexical_cast<string>(i);
            Parameters* p_x2hb = model->add_parameters({ long(hidden_dim) }, iscale, i_name);
            vector<Parameters*> ps = { p_x2h, p_x2hb };
            params.push_back(ps);
            layer_input_dim = hidden_dim;
        }
    }

    void DNNBuilder::new_graph_impl(ComputationGraph& cg) {
        param_vars.clear();
        for (unsigned i = 0; i < layers; ++i) {
            Parameters* p_x2h = params[i][X2H];
            Parameters* p_x2hb = params[i][X2HB];
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

            //    Expression y = affine_transform({ vars[2], vars[0], x });
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

} // namespace cnn
