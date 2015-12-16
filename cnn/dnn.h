#ifndef _DNN_H
#define _DNN_H

/**
this uses RNN interface but actually is deep network
*/
#include "cnn/rnn.h"

using namespace cnn::expr;

namespace cnn {

    class DNNBuilder  {
    public:
        DNNBuilder() { dparallel = 1; }
        explicit DNNBuilder(unsigned layers,
            unsigned input_dim,
            unsigned hidden_dim,
            Model* model,
            float i_scale = 1.0);
        /// for parameter sharing 
        DNNBuilder(const DNNBuilder& ref)
        {
            input_dims = ref.input_dims;
            params = ref.params;
            param_vars = ref.param_vars;
            layers = input_dims.size();
            dparallel = ref.dparallel;
        }

        ~DNNBuilder() {}

    protected:
        virtual void new_graph_impl(ComputationGraph& cg);
        virtual Expression add_input_impl(const Expression& x);

    public:
        Expression back() const { return h.back(); }
        std::vector<Expression> final_h() const { return h; }
        std::vector<Expression> final_s() const { return final_h(); }
        void copy(const DNNBuilder & params);

        unsigned num_h0_components() const { return layers; }

        void set_data_in_parallel(int n);
        int data_in_parallel() const { return dparallel; }

    public:
        void display(ComputationGraph& cg);

    public:
        // add another timestep by reading in the variable x
        // return the hidden representation of the deepest layer
        Expression add_input(const Expression& x) {
            return add_input_impl(x);
        }

        // call this to reset the builder when you are working with a newly
        // created ComputationGraph object
        void new_graph(ComputationGraph& cg) {
            new_graph_impl(cg);
        }

        // Reset for new sequence
        // call this before add_input and after new_graph,
        // when starting a new sequence on the same hypergraph.
        // h_0 is used to initialize hidden layers at timestep 0 to given values
        void start_new_sequence(const std::vector<Expression>& h_0 = {}) {
        }


    private:

        unsigned layers;  /// number of layers

        std::vector<unsigned> input_dims;  /// input dimension at each layer

        int dparallel;

        /// for parameters
        // first index is layer, then ...
        std::vector<std::vector<Parameters*>> params;
        // first index is layer, then ...
        std::vector<std::vector<Expression>> param_vars;

        std::vector<Expression> h;

        std::vector<std::vector<Expression>> biases;
    };
};

#endif
