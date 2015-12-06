#ifndef _EVAL_PROC_H
#define _EVAL_PROC_H

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/cnn-helper.h"
#include "ext/dialogue/attention_with_intention.h"
#include "cnn/data-util.h"
#include "cnn/grad-check.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_wiarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_woarchive.hpp>
#include <boost/archive/codecvt_null.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;

/**
The higher level evaluation process
Proc: a decode process that can output decode results
*/
template <class Proc>
class EvaluateProcess{
private:
    map<int, Expression> eWordEmbedding;
    map<int, vector<cnn::real>> vWordEmbedding;

public:
    EvaluateProcess()
    {
    }

    void readEmbedding(const string& embedding_fn, Dict& sd);
    void emb2expression(ComputationGraph& cg);

    Expression score(const vector<int>& ref, const vector<int>& res, ComputationGraph& cg);
    cnn::real score(Expression er, ComputationGraph& cg);

    int scoreInEmbeddingSpace(Proc &am, Dialogue &v_v_dialogues, Dict& td, double &dloss, double & dchars_s, double & dchars_t);
};

template<class Proc>
void EvaluateProcess<Proc>::readEmbedding(const string& embedding_fn, Dict& sd)
{
    ifstream in(embedding_fn);
    string line;

    while (getline(in, line)) {

        int wrd_idx;

        vector<cnn::real> iv = read_embedding(line, sd, wrd_idx);
        if (wrd_idx >= 0)
            vWordEmbedding[wrd_idx] = iv;
    }

    in.close();
}

template<class Proc>
void EvaluateProcess<Proc>::emb2expression(ComputationGraph& cg)
{
    eWordEmbedding.clear();
    for (auto& p : vWordEmbdding)
    {
        Expression x = vec2exp(p.second, cg);
        eWordEmbedding[p.first] = x;
    }
}

template<class Proc>
Expression EvaluateProcess<Proc>::score(const vector<int>& ref, const vector<int>& res, ComputationGraph& cg)
{
    vector<Expression> vref;
    for (auto & p : ref)
    {
        if (vWordEmbedding.find(p) != vWordEmbedding.end())
        {
            Expression x = vec2exp(vWordEmbedding[p], cg);
            vref.push_back(x);
        }
    }

    vector<Expression> vres;
    for (auto & p : res)
    {
        if (vWordEmbedding.find(p) != vWordEmbedding.end())
        {
            Expression x = vec2exp(vWordEmbedding[p], cg);
            vres.push_back(x);
        }
    }

    Expression sim_n = squared_distance(average(vref), average(vres));
    return sim_n;
}

template<class Proc>
cnn::real EvaluateProcess<Proc>::score(Expression er, ComputationGraph& cg)
{
    return get_value(er, cg)[0];
}

template <class AM_t>
int EvaluateProcess<AM_t>::scoreInEmbeddingSpace(AM_t &am, Dialogue &v_v_dialogues, Dict& td, double &dloss, double & dchars_s, double & dchars_t)
{
    size_t i_turns = 0;
    size_t turn_id = 0;
    vector<int> prv_response;
    SentencePair prv_turn;
    ComputationGraph cg;
    vector<int> decode_output;
    vector<cnn::real> vscore;

    for (auto turn : v_v_dialogues)
    {
        if (turn_id == 0)
        {
            decode_output = am.decode(turn.first, cg, td);
        }
        else
        {
            //            am.build_graph(prv_response, turn.first, cg, td);
            decode_output = am.decode(prv_turn.second, turn.first, cg, td);
        }

        Expression esc = score(turn.second, decode_output, cg);
        cnn::real isc = score(esc, cg);

        vscore.push_back(isc);

        prv_response = decode_output;
        prv_turn = turn;

        dloss += vscore.back();
        dchars_s += turn.first.size() - 1;
        dchars_t += turn.second.size() - 1;

        turn_id++;
        i_turns++;
    }
    return turn_id;
}



#endif
