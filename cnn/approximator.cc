#include "cnn/approximator.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "cnn/nodes.h"
#include "cnn/expr-xtra.h"
#include <fstream>

using namespace std;
using namespace cnn::expr;
using namespace cnn;

namespace cnn {

    enum { X2C = 0, X2CB};

    void ClsBasedBuilder::display(ComputationGraph& cg) {
        display_value(i_cls, cg);
        display_value(i_cls_bias, cg);
        for (unsigned i = 0; i < ncls; ++i) {
            const std::vector<Expression>& vars = param_vars[i];
            for (size_t i = 0; i < vars.size(); i++)
                display_value(vars[i], cg);
        }
    }

    ClsBasedBuilder::ClsBasedBuilder(
        const unsigned int input_dim,
        const vector<int>& cls2nbrwords, /// #words for each class, class starts from 0
        const vector<long>& word2cls,
        const vector<long>& dict_wrd_id2within_class_id,
        Model& model,
        cnn::real iscale,
        string name) : input_dim(input_dim),
        clssize(cls2nbrwords), word2cls(word2cls), dict_wrd_id2within_class_id(dict_wrd_id2within_class_id)
    {
        unsigned int n_cls = clssize.size();
        p_cls = model.add_parameters({ n_cls, input_dim }, iscale, name + " to cls");
        p_cls_bias = model.add_parameters({ n_cls }, iscale, name + " to cls bias");
        for (size_t id = 0; id < n_cls; id++)
        {
            unsigned int  clssize = cls2nbrwords[id];
            p_R.push_back(model.add_parameters({ clssize, input_dim }, iscale, name + " to cls " + boost::lexical_cast<string>(id)));
            p_bias.push_back(model.add_parameters({ clssize }, iscale, name + " to cls bias " + boost::lexical_cast<string>(id)));
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
        errors.clear();
        set_data_in_parallel(1);
    }

    void ClsBasedBuilder::set_data_in_parallel(int n)
    {
        dparallel = n;

        errors.clear();
        errors.resize(n);
    }

    Expression ClsBasedBuilder::add_input_impl(const Expression &in, long target_wordid, unsigned uttid) {
        int cls_id = word2cls[target_wordid];
        Expression newin = reshape(in, { input_dim * dparallel });
        Expression x = pickrange(newin, uttid * input_dim, (uttid + 1)*input_dim);
        vector<Expression> param = param_vars[cls_id];
        Expression i_r_t = param[X2CB] + param[X2C] * x;
        Expression i_c_t = i_cls_bias + i_cls * x;
        Expression i_err_cls = pick(log_softmax(i_c_t), cls_id);
        Expression i_err_prb = pick(log_softmax(i_r_t), dict_wrd_id2within_class_id[target_wordid]);
        errors[uttid].push_back(-(i_err_cls + i_err_prb));

        return errors[uttid].back();
    }

    vector<cnn::real> ClsBasedBuilder::respond(const Expression &in, ComputationGraph& cg) 
    {
        Expression cls_log_prob = log_softmax(i_cls * in + i_cls_bias);
        vector<Expression> within_cls_log_prob;
        for (int cli = 0; cli < clssize.size(); cli++)
        {
            vector<Expression> param = param_vars[cli];
            Expression i_r_t = param[X2CB] + param[X2C] * in;
            within_cls_log_prob.push_back(log_softmax(i_r_t));
        }
        auto cls_dist = get_value(cls_log_prob, cg);
        vector<cnn::real> dist;
        for (int cli = 0; cli < clssize.size(); cli++){
            auto within_cls_dist = get_value(within_cls_log_prob[cli], cg);
            vector<cnn::real> vtmp(clssize[cli], cls_dist[cli]);
            std::transform(within_cls_dist.begin(), within_cls_dist.end(), vtmp.begin(), within_cls_dist.begin(), std::plus<cnn::real>());
            dist.insert(dist.end(), within_cls_dist.begin(), within_cls_dist.end());
        }

        return dist;
    }

    void ClsBasedBuilder::copy(const ClsBasedBuilder& othercls) {
        const ClsBasedBuilder& clsref = (const ClsBasedBuilder&)othercls;
        assert(p_R.size() == clsref.p_R.size());
        assert(p_bias.size() == clsref.p_bias.size());
        for (size_t i = 0; i < clsref.p_R.size(); ++i) {
            p_R[i]->copy(*clsref.p_R[i]);
            p_bias[i]->copy(*clsref.p_bias[i]);
        }
        p_cls->copy(*clsref.p_cls);
        p_cls_bias->copy(*clsref.p_cls_bias);
    }


    /**
    dict_wrd_id2within_class_id : the dictionary word id to the id inside a class
    */
    void ClsBasedBuilder::load_word2cls_fn(string word2clsfn, Dict& sd, std::vector<long>& wrd2cls, std::vector<long>& dict_wrd_id2within_class_id, std::vector<int> & cls2size)
    {
        ifstream in(word2clsfn);
        string line;
        /// go through one line at a time to match dictionay
        /// it is possible that the word clustering may have different words from the dictionary, 
        /// so go through one more to add missing words into the dictionary
        vector<string> missing_words;
        while (getline(in, line)) {
            std::istringstream in(line);
            std::string word;

            in >> word;
            if (sd.Contains(word) == false)
                missing_words.push_back(word);
            sd.Convert(word, true);
        }
        in.close();

        in.open(word2clsfn);
        wrd2cls.resize(sd.size());
        dict_wrd_id2within_class_id.resize(sd.size(), -1);
        map<int, int> cls2acccnt; /// the count for each class so far
        while (getline(in, line)) {

            std::istringstream in(line);
            std::string word;
            string cls;

            in >> word;
            in >> cls;

            int icls = boost::lexical_cast<int>(cls)-1;
            int wridx = -1;
            if (sd.Contains(word))
            {
                wridx = sd.Convert(word, true);

                wrd2cls[wridx] = icls;
                if (cls2acccnt.find(icls) == cls2acccnt.end())
                    cls2acccnt[icls] = 1;
                else
                    cls2acccnt[icls] += 1;
                dict_wrd_id2within_class_id[wridx] = cls2acccnt[icls] - 1;
            }
        }
        in.close();

        for (auto&p : sd.GetWordList())
        {
            long wd = sd.Convert(p);
            if (wrd2cls[wd] == -1)
                throw("check word clustering procedure as there are words in dictionary that are not clustered"); 
        }

        cls2size.clear();
        for (int i = 0; i < cls2acccnt.size(); i++)
        {
            cls2size.push_back(cls2acccnt[i]);
        }
    }


} // namespace cnn
