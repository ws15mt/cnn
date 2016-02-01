#ifndef _APPROXIMATOR_H
#define _APPROXIMATOR_H

/**
this is for approximating criterion.
1) class-based objective
To-do: NCE
*/
#include <string>
#include <vector>
#include "cnn/model.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/dict.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

    class ClsBasedBuilder{
    public:
        ClsBasedBuilder() { dparallel = 1; }
        explicit ClsBasedBuilder(
            const unsigned int input_dim,
            const vector<int>& cls2nbrwords, /// #words for each class, class starts from 0
            const vector<long>& word2cls,
            const vector<long>& dict_wrd_id2within_class_id,
            Model& model,
            cnn::real iscale,
            string name = "");
        ClsBasedBuilder(const ClsBasedBuilder& ref)
        {
            p_R = ref.p_R;
            p_bias = ref.p_bias;
            p_cls = ref.p_cls;
            p_cls_bias = ref.p_cls_bias;
            ncls = ref.ncls;
            clssize = ref.clssize;
            word2cls = ref.word2cls;
            dict_wrd_id2within_class_id = ref.dict_wrd_id2within_class_id;
            dparallel = ref.dparallel;
        }

        ~ClsBasedBuilder() {}

    protected:
        void new_graph_impl(ComputationGraph& cg);
        virtual Expression add_input_impl(const Expression& x, long target_wordid, unsigned uttid);

    public:
        vector<Expression> back() const { return errors.back(); }
        void copy(const ClsBasedBuilder& params);

        void set_data_in_parallel(int n);
        int data_in_parallel() const { return dparallel; }

    public:
        void display(ComputationGraph& cg);

    public:
        // add another timestep by reading in the variable x
        // targetid is a vector of training target. for target dont want to have error signals, set the corresponding element to a negative value
        // return the errors
        vector<Expression> add_input(const Expression& x, vector<long> targetid) {
            vector<Expression> err; 
            unsigned id = 0;
            for (auto& p : targetid)
            {
                if (p >= 0)
                    err.push_back(add_input_impl(x, p, id));
                id++;
            }
            return err;
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
        void start_new_sequence() {
        }

        vector<cnn::real> respond(const Expression &in, ComputationGraph& cg);

    public:
        void load_word2cls_fn(string word2clsfn, Dict& sd, std::vector<long>& wrd2cls, std::vector<long>& dict_wrd_id2within_class_id, std::vector<int> & cls2size);
    protected:

        vector<Parameters*> p_R;
        vector<Parameters*> p_bias;
        Parameters* p_cls, *p_cls_bias;
        Expression i_cls, i_cls_bias;
        unsigned int ncls;
        unsigned input_dim;
        vector<int> clssize;
        vector<long> word2cls;
        vector<long> dict_wrd_id2within_class_id;
        
        int dparallel;

        std::vector<std::vector<Expression>> param_vars;

        vector<vector<Expression>> errors; /// [nutt][vector<error>] 
    };

};

#endif
