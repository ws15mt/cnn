#ifndef _TRAIN_PROC_H
#define _TRAIN_PROC_H

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

#ifdef CUDA
#define NBR_DEV_PARALLEL_UTTS 2
#else
#define NBR_DEV_PARALLEL_UTTS 10
#endif

#define LEVENSHTEIN_THRESHOLD 5

unsigned LAYERS = 2;
unsigned HIDDEN_DIM = 50;  // 1024
unsigned ALIGN_DIM = 25;  // 1024
unsigned VOCAB_SIZE_SRC = 0;
long nparallel = -1;
long mbsize = -1;
unsigned min_diag_id = 0;
size_t g_train_on_turns = 1; 

cnn::Dict sd;

int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
int verbose;
int beam_search_decode;
float lambda = 1e-6;
int repnumber;

Sentence prv_response;

NumTurn2DialogId training_numturn2did;
NumTurn2DialogId devel_numturn2did;
NumTurn2DialogId test_numturn2did;

/**
The higher level training process
*/
template <class Proc>
class TrainProcess{
public:
    TrainProcess()
    {
    }

    void prt_model_info(size_t LAYERS, size_t VOCAB_SIZE_SRC, const vector<unsigned>& dims, size_t nreplicate, size_t decoder_additiona_input_to, size_t mem_slots, cnn::real scale);

    void batch_train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs, int nparallel);
    void supervised_pretrain(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, float target_ppl, int min_diag_id,
        bool bcharlevel = false, bool nosplitdialogue = false);
    void train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs, int min_diag_id,
        bool bcharlevel = false, bool nosplitdialogue = false);
    void test(Model &model, Proc &am, Corpus &devel, string out_file, Dict & td, NumTurn2DialogId& test_corpusinfo, const string& score_embedding_fn);
    void dialogue(Model &model, Proc &am, string out_file, Dict & td);

    void collect_sample_responses(Proc& am, Corpus &training);

    void nosegmental_forward_backward(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt,
        double &dloss, double & dchars_s, double & dchars_t, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr);
    void segmental_forward_backward(Model &model, Proc &am, const PDialogue &v_v_dialogues, Trainer &sgd, bool bupdate, int nutt,
        double &dloss, double & dchars_s, double & dchars_t);

    cnn::real smoothed_ppl(cnn::real curPPL);
    void reset_smoothed_ppl(){
        ppl_hist.clear();
    }

private:
    vector<cnn::real> ppl_hist;

};

/**
this is fake experiment as the user side is known and supposedly respond correctly to the agent side
*/
template <class AM_t>
void TrainProcess<AM_t>::test(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & td, NumTurn2DialogId& test_corpusinfo,
    const string& score_embedding_fn)
{
    unsigned lines = 0;
    float dloss = 0;
    float dchars_s  = 0;
    float dchars_t = 0;

    ofstream of(out_file);

    unsigned si = devel.size(); /// number of dialgoues in training

    Timer iteration("completed in");
    double ddloss = 0;
    double ddchars_s = 0;
    double ddchars_t = 0;

    {
        vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
        size_t id_stt_diag_id = 0;
        PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
        vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);
        size_t ndutt = id_sel_idx.size();

        if (verbose)
        {
            cerr << "selected " << ndutt << " :  ";
            for (auto p : id_sel_idx)
                cerr << p << " ";
            cerr << endl;
        }

        while (ndutt > 0)
        {
            nosegmental_forward_backward(model, am, vd_dialogues, ndutt, ddloss, ddchars_s, ddchars_t, true);

            id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, test_corpusinfo);
            ndutt = id_sel_idx.size();

            if (verbose)
            {
                cerr << "selected " << ndutt << " :  ";
                for (auto p : id_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }
        }
    }

    cerr << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << ' ';
    of << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << endl; 

    /// if report score in embedding space
    if (score_embedding_fn.size() > 0)
    {
        EvaluateProcess<AM_t> * ptr_evaluate = new EvaluateProcess<AM_t>();
        ptr_evaluate->readEmbedding(score_embedding_fn, td);

        double emb_loss = 0;
        double emb_chars_s = 0;
        double emb_chars_t = 0;
        double turns = 0;
        for (auto & diag : devel)
        {
            turns += ptr_evaluate->scoreInEmbeddingSpace(am, diag, td, emb_loss, emb_chars_s, emb_chars_t);
        }
        cerr << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] word embedding loss = " << (emb_loss / turns) << " ppl=" << exp(emb_loss / turns) << ' ';
        of << "\n***Test [lines =" << lines << " out of total " << devel.size() << " lines ] word embedding loss = " << (emb_loss / turns) << " ppl=" << exp(emb_loss / turns) << endl;

        delete ptr_evaluate;
    }

    of.close();
}

template <class AM_t>
void TrainProcess<AM_t>::dialogue(Model &model, AM_t &am, string out_file, Dict & td)
{
    string shuman;
    ofstream of(out_file);
    unsigned lines = 0;

    int d_idx = 0;
    while (1){
        cout << "please start dialogue with the agent. you can end this dialogue by typing exit " << endl;

        size_t t_idx = 0;
        vector<int> decode_output;
        vector<int> shuman_input;
        Sentence prv_response;
        ComputationGraph cg;
        while (1){
#ifdef INPUT_UTF8
            std::getline(wcin, shuman);
            if (shuman.find(L"exit") == 0)
                break;
#else
            std::getline(cin, shuman);
            if (shuman.find("exit") == 0)
                break;
#endif

            convertHumanQuery(shuman, shuman_input, td);

            if (t_idx == 0)
                decode_output = am.decode(shuman_input, cg, td);
            else
                decode_output = am.decode(prv_response, shuman_input, cg, td);

            of << "res ||| " << d_idx << " ||| " << t_idx << " ||| ";
            for (auto pp : shuman_input)
            {
                of << td.Convert(pp) << " ";
            }
            of << " ||| ";

            for (auto pp : decode_output)
            {
                of << td.Convert(pp) << " ";
            }
            of << endl;

            cout << "Agent: ";
            for (auto pp : decode_output)
            {
                cout << td.Convert(pp) << " ";
            }
            cout << endl;

            prv_response = decode_output;
            t_idx++;
        }
        d_idx++;
        of << endl;
    }

    of.close();
}

template <class AM_t>
void TrainProcess<AM_t>::nosegmental_forward_backward(Model &model, AM_t &am, PDialogue &v_v_dialogues, int nutt,
    double &dloss, double & dchars_s, double & dchars_t, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr)
{
    size_t turn_id = init_turn_id;
    size_t i_turns = 0;
    PTurn prv_turn;

    ComputationGraph cg;
    if (resetmodel)
    {
        am.reset();
    }

    for (auto turn : v_v_dialogues)
    {
        if (turn_id == 0)
        {
            am.build_graph(turn, cg);
        }
        else
        {
            am.build_graph(prv_turn, turn, cg);
        }

        cg.incremental_forward();
        //            CheckGrad(model, cg);

        prv_turn = turn;
        turn_id++;
        i_turns++;
    }

    dloss += as_scalar(cg.get_value(am.s2txent.i));

    dchars_s += am.swords;
    dchars_t += am.twords;

    if (sgd != nullptr)
    {
        cg.backward();
        sgd->update(am.twords);
    }
}

/**
Deal with very long input
Use multiple segments back-propagation
1) first pass to do forward prop. save the last state of one segment. 
2) delete graph of this segment, 
3) do forward propagation on the following segment, with initial state from the last segment, which has been recorded
4) do 1-3) untill all segments are processed
5) check if forward prop states are there. if not, do forward prop on this segment, with the initial state from the last segment
6) do backward prop on this segment, with the last state errors from the future segment. 
7) do 5-6) until all segments are processed
8) do sgd update
*/
template <class AM_t>
void TrainProcess<AM_t>::segmental_forward_backward(Model &model, AM_t &am, const PDialogue &v_v_dialogues, Trainer &sgd, bool bupdate, int nutt,
    double &dloss, double & dchars_s, double & dchars_t)
{
    size_t seg_len = 1; /// one turn 
    int turn_id = 0;
    size_t seg_id = 0;
    PTurn prv_turn;

    PDialogue vpd;
    vector<PDialogue> v_vpd;
    vector<size_t> seg2turn; 

    am.reset();

    /// first pass 
    for (auto v_p : v_v_dialogues)
    {
        vpd.push_back(v_p);
        if (turn_id % seg_len == 0)
        {
            v_vpd.push_back(vpd);
            nosegmental_forward_backward(model, am, vpd, nutt, dloss, dchars_s, dchars_t, seg_id == 0, turn_id);
            seg2turn.push_back(turn_id);
            vpd.clear();
            seg_id++;
        }
        turn_id++;
    }
    if (vpd.size() > 0)
    {
        v_vpd.push_back(vpd);
        nosegmental_forward_backward(model, am, vpd, nutt, dloss, dchars_s, dchars_t, seg_id == 0, turn_id);
        seg_id++;
        seg2turn.push_back(turn_id);
    }

    /// second pass to update parameters
    for (auto v_p : boost::adaptors::reverse(v_vpd))
    {
        seg_id--;
        nosegmental_forward_backward(model, am, v_p, bupdate, nutt, dloss, dchars_s, dchars_t, false, seg2turn[seg_id]);
        if (bupdate)
        {
            cg.backward();
            sgd.update(nutt * v_p.size());
        }
    }
}

/* the following does mutiple sentences per minibatch 
but I comment it out 
*/
template <class AM_t>
void TrainProcess<AM_t>::batch_train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, int nparallel)
{
    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    ofstream out(out_file, ofstream::out);
    boost::archive::text_oarchive oa(out);
    oa << model;
    out.close();

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (sgd.epoch < max_epochs) {
        Timer iteration("completed in");
        double dloss = 0;
        double dchars_s = 0;
        double dchars_t = 0;
        double dchars_tt = 0;

        for (unsigned iter = 0; iter < report_every_i;) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd.update_epoch(); }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                /// shuffle number of turns
                shuffle(training_numturn2did.vNumTurns.begin(), training_numturn2did.vNumTurns.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(training.size(), false);
                for (auto p : training_numturn2did.mapNumTurn2DialogId){
                    /// shuffle dailogues with the same number of turns
                    random_shuffle(p.second.begin(), p.second.end());
                }
                v_selected.assign(training.size(), false);  
            }

            {
                Dialogue prv_turn;
                size_t turn_id = 0;
                PDialogue v_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
                vector<int> i_sel_idx = get_same_length_dialogues(training, nparallel, i_stt_diag_id, v_selected, v_dialogues, training_numturn2did);
                size_t nutt = i_sel_idx.size();

                if (verbose)
                {
                    cerr << "selected " << nutt << " :  ";
                    for (auto p : i_sel_idx)
                        cerr << p << " ";
                    cerr << endl;
                }

                nosegmental_forward_backward(model, am, v_dialogues, nutt, dloss, dchars_s, dchars_t, true, 0, &sgd);

                si+=nutt;
                lines+=nutt;
                iter += nutt;

            }
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

        // show score on dev data?
        report++;
        if (floor(sgd.epoch) != prv_epoch || report % dev_every_i_reports == 0  || fmod(lines, (double)training.size()) == 0.0) {
            double ddloss = 0;
            double ddchars_s = 0;
            double ddchars_t = 0;

            {
                vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
                size_t id_stt_diag_id = 0;
                PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
                vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                size_t ndutt = id_sel_idx.size();

                if (verbose)
                {
                    cerr << "selected " << ndutt << " :  ";
                    for (auto p : id_sel_idx)
                        cerr << p << " ";
                    cerr << endl;
                }

                while (ndutt > 0)
                {
                    nosegmental_forward_backward(model, am, vd_dialogues, ndutt, ddloss, ddchars_s, ddchars_t, true);

                    id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                    ndutt = id_sel_idx.size();

                    if (verbose)
                    {
                        cerr << "selected " << ndutt << " :  ";
                        for (auto p : id_sel_idx)
                            cerr << p << " ";
                        cerr << endl;
                    }
                }
            }
            if (ddloss < best) {
                best = ddloss;
                ofstream out(out_file , ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else if (ddloss > best * 1.05){
                sgd.eta *= 0.5; /// reduce learning rate
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);
    }
}

/**
@bcharlevel : true if character output; default false.
*/
template <class AM_t>
void TrainProcess<AM_t>::train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, int min_diag_id,
    bool bcharlevel = false, bool nosplitdialogue = false)
{
    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    boost::mt19937 rng;                 // produces randomness out of thin air

    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    ofstream out(out_file, ofstream::out);
    boost::archive::text_oarchive oa(out);
    oa << model;
    out.close();

    reset_smoothed_ppl();

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (sgd.epoch < max_epochs) {
        Timer iteration("completed in");
        double dloss = 0;
        double dchars_s = 0;
        double dchars_t = 0;
        double dchars_tt = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd.update_epoch(); }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(training.size(), false);
            }

            // build graph for this instance
            auto& spair = training[order[si % order.size()] + min_diag_id];

            if (verbose)
                cerr << "diag = " << order[si % order.size()] + min_diag_id << endl;

            /// find portion to train
            bool b_trained = false;
            // see random number distributions
            auto rng = std::bind(std::uniform_int_distribution<int>(0, spair.size() - 1), *rndeng);
            int i_turn_to_train = rng();
            if (nosplitdialogue)
                i_turn_to_train = 99999;

            vector<SentencePair> prv_turn;
            size_t turn_id = 0;
            
            size_t i_init_turn = 0;

            /// train on two segments of a dialogue
            do{
                ComputationGraph cg;

                if (i_init_turn > 0)
                    am.assign_cxt(cg, 1);
                for (size_t t = i_init_turn; t <= std::min(i_init_turn + i_turn_to_train, spair.size()-1); t++)
                {
                    SentencePair turn = spair[t];
                    vector<SentencePair> i_turn(1, turn);
                    if (turn_id == 0)
                    {
                        am.build_graph(i_turn, cg);
                    }
                    else
                    {
                        am.build_graph(prv_turn, i_turn, cg);
                    }
                    cg.incremental_forward();
                    turn_id++;

                    if (verbose)
                    {
                        display_value(am.s2txent, cg);
                        float tcxtent = as_scalar(cg.get_value(am.s2txent));
                        cerr << "xent = " << tcxtent << " nobs = " << am.twords << " PPL = " << exp(tcxtent / am.twords) << endl;
                    }

                    prv_turn = i_turn;
                    if (t == i_init_turn + i_turn_to_train || (t == spair.size() - 1)){

                        dloss += as_scalar(cg.get_value(am.s2txent.i));

                        dchars_s += am.swords;
                        dchars_t += am.twords;

                        cg.backward();
                        sgd.update(am.twords);

                        am.serialise_cxt(cg);
                        i_init_turn = t + 1;
                        i_turn_to_train = spair.size() - i_init_turn;
                        break;
                    }
                }
            } while (i_init_turn < spair.size());

            if (iter == report_every_i - 1)
                am.respond(spair, sd, bcharlevel);

            ++si;
            lines++;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';


        // show score on dev data?
        report++;
        if (floor(sgd.epoch) != prv_epoch || report % dev_every_i_reports == 0 || fmod(lines, (double)training.size()) == 0.0) {
            double ddloss = 0;
            double ddchars_s = 0;
            double ddchars_t = 0;

            {
                vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
                size_t id_stt_diag_id = 0;
                PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
                vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                size_t ndutt = id_sel_idx.size();

                if (verbose)
                {
                    cerr << "selected " << ndutt << " :  ";
                    for (auto p : id_sel_idx)
                        cerr << p << " ";
                    cerr << endl;
                }

                while (ndutt > 0)
                {
                    nosegmental_forward_backward(model, am, vd_dialogues, ndutt, ddloss, ddchars_s, ddchars_t, true);

                    id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                    ndutt = id_sel_idx.size();

                    if (verbose)
                    {
                        cerr << "selected " << ndutt << " :  ";
                        for (auto p : id_sel_idx)
                            cerr << p << " ";
                        cerr << endl;
                    }
                }
            }
            ddloss = smoothed_ppl(ddloss);
            if (ddloss < best) {
                best = ddloss;
                ofstream out(out_file, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else{
                sgd.eta0 *= 0.5; /// reduce learning rate
                sgd.eta *= 0.5; /// reduce learning rate
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);
    }
}

/**
collect sample responses
*/
template <class AM_t>
void TrainProcess<AM_t>::collect_sample_responses(AM_t& am, Corpus &training)
{
    am.clear_candidates();
    for (auto & ds: training){
        vector<SentencePair> prv_turn;
        size_t turn_id = 0;

        for (auto& spair : ds){
            SentencePair turn = spair;
            am.collect_candidates(spair.second);
        }
    }
}

/// smooth PPL on three points with 0.9 forgetting factor
template<class AM_t>
cnn::real TrainProcess<AM_t>::smoothed_ppl(cnn::real curPPL)
{
    if (ppl_hist.size() == 0)
        ppl_hist.resize(3, curPPL);
    ppl_hist.push_back(curPPL);
    if (ppl_hist.size() > 3)
        ppl_hist.erase(ppl_hist.begin());

    cnn::real finPPL = 0;
    size_t k = 1;
    for (auto p : ppl_hist)
    {
        finPPL += pow(0.9, 3 - k) * p;
        k++;
    }
    return finPPL;
}

/**
overly pre-train models on small subset of the data 
*/
template <class AM_t>
void TrainProcess<AM_t>::supervised_pretrain(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, cnn::real target_ppl, int min_diag_id,
    bool bcharlevel = false, bool nosplitdialogue = false)
{
    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    boost::mt19937 rng;                 // produces randomness out of thin air

    reset_smoothed_ppl();

    size_t sample_step = 100;
    size_t maxepoch = sample_step * 10; /// no point of using more than 100 epochs, which correspond to use full data with 10 epochs for pre-train
    vector<unsigned> order(training.size()/sample_step);
    size_t k = 0;
    for (unsigned i = 0; i < training.size(); i += sample_step)
    {
        if (k < order.size())
            order[k++] = i;
        else
            break;
    }

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    ofstream out(out_file, ofstream::out);
    boost::archive::text_oarchive oa(out);
    oa << model;
    out.close();

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (best > target_ppl && sgd.epoch < maxepoch) {
        Timer iteration("completed in");
        double dloss = 0;
        double dchars_s = 0;
        double dchars_t = 0;
        double dchars_tt = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd.update_epoch(); }
            }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
                i_stt_diag_id = 0;
                v_selected = vector<bool>(order.size(), false);
            }

            // build graph for this instance
            auto& spair = training[order[si % order.size()] + min_diag_id];

            if (verbose)
                cerr << "diag = " << order[si % order.size()] + min_diag_id << endl;

            /// find portion to train
            bool b_trained = false;
            // see random number distributions
            auto rng = std::bind(std::uniform_int_distribution<int>(0, spair.size() - 1), *rndeng);
            int i_turn_to_train = rng();
            if (nosplitdialogue)
                i_turn_to_train = 99999;

            vector<SentencePair> prv_turn;
            size_t turn_id = 0;

            size_t i_init_turn = 0;

            /// train on two segments of a dialogue
            do{
                ComputationGraph cg;

                if (i_init_turn > 0)
                    am.assign_cxt(cg, 1);
                for (size_t t = i_init_turn; t <= std::min(i_init_turn + i_turn_to_train, spair.size() - 1); t++)
                {
                    SentencePair turn = spair[t];
                    vector<SentencePair> i_turn(1, turn);
                    if (turn_id == 0)
                    {
                        am.build_graph(i_turn, cg);
                    }
                    else
                    {
                        am.build_graph(prv_turn, i_turn, cg);
                    }
                    cg.incremental_forward();
                    turn_id++;

                    if (verbose)
                    {
                        display_value(am.s2txent, cg);
                        float tcxtent = as_scalar(cg.get_value(am.s2txent));
                        cerr << "xent = " << tcxtent << " nobs = " << am.twords << " PPL = " << exp(tcxtent / am.twords) << endl;
                    }

                    prv_turn = i_turn;
                    if (t == i_init_turn + i_turn_to_train || (t == spair.size() - 1)){

                        dloss += as_scalar(cg.get_value(am.s2txent.i));

                        dchars_s += am.swords;
                        dchars_t += am.twords;

                        cg.backward();
                        sgd.update(am.twords);

                        am.serialise_cxt(cg);
                        i_init_turn = t + 1;
                        i_turn_to_train = spair.size() - i_init_turn;
                        break;
                    }
                }
            } while (i_init_turn < spair.size());

            if (iter == report_every_i - 1)
                am.respond(spair, sd, bcharlevel);

            ++si;
            lines++;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (double)order.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

        prv_epoch = floor(sgd.epoch);

        cnn::real i_ppl = smoothed_ppl(exp(dloss / dchars_t));
        if (best > i_ppl)
        {
            best = i_ppl;
        }
        else
        {
            sgd.eta0 *= 0.5;
            sgd.eta *= 0.5;
        }
        if (sgd.eta < 1e-10)
        {
            cerr << "SGD stepsize is too small to update models" << endl;
            break;
        }
    }

    ofstream out2(out_file, ofstream::out);
    boost::archive::text_oarchive oa2(out2);
    oa2 << model;
    out2.close();

    ofstream out3(out_file + ".pretrained", ofstream::out);
    boost::archive::text_oarchive oa3(out3);
    oa3 << model;
    out3.close();

}

template<class rnn_t, class TrainProc>
void prt_model_info(size_t LAYERS, size_t VOCAB_SIZE_SRC, const vector<unsigned>& dims, size_t nreplicate, size_t decoder_additiona_input_to, size_t mem_slots, cnn::real scale)
{
    cerr << "layer = " << LAYERS << endl; 
    cerr << "vocab size = " << VOCAB_SIZE_SRC << endl;
    cerr << "dims = "; 
    for (auto & p : dims)
    {
        cerr << " " << p;
    }
    cerr << endl;
    cerr << "nreplicate = " << nreplicate << endl;
    cerr << "decoder_additional_input_to = " << decoder_additiona_input_to << endl;
    cerr << "mem_slots = " << mem_slots << endl;
    cerr << "scale = " << scale << endl;
}

template <class rnn_t, class TrainProc>
int main_body(variables_map vm, size_t nreplicate= 0, size_t decoder_additiona_input_to = 0, size_t mem_slots = MEM_SIZE)
{
#ifdef INPUT_UTF8
    kSRC_SOS = sd.Convert(L"<s>");
    kSRC_EOS = sd.Convert(L"</s>");
#else
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
#endif
    verbose = vm.count("verbose");
    g_train_on_turns = vm["turns"].as<int>();

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel, testcorpus;
    string line;

    TrainProc  * ptrTrainer = nullptr;

    if (vm.count("readdict"))
    {
        string fname = vm["readdict"].as<string>();
#ifdef INPUT_UTF8
        wifstream in(fname, wifstream::in);
        boost::archive::text_wiarchive ia(in);
#else
        ifstream in(fname, ifstream::in);
        boost::archive::text_iarchive ia(in);
#endif
        ia >> sd;
        sd.Freeze();
    }

    if (vm.count("train") > 0)
    {
        cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
        training = read_corpus(vm["train"].as<string>(), min_diag_id, sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), vm.count("appendBOSEOS")> 0,
            vm.count("charlevel") > 0);
        sd.Freeze(); // no new word types allowed

        training_numturn2did = get_numturn2dialid(training);

        if (vm.count("writedict"))
        {
            string fname = vm["writedict"].as<string>();
#ifdef INPUT_UTF8
            wstring wfname;
            wfname.assign(fname.begin(), fname.end());
            wofstream ofs(wfname);
            boost::archive::text_woarchive oa(ofs);
#else
            ofstream on(fname);
            boost::archive::text_oarchive oa(on);
#endif
            oa << sd;
        }
    }
    else
    {
        if (vm.count("readdict") == 0)
        {
            cerr << "must have either training corpus or dictionary" << endl;
            abort();
        }
    }

    LAYERS = vm["layers"].as<int>();
    HIDDEN_DIM = vm["hidden"].as<int>();
    ALIGN_DIM = vm["align"].as<int>();

    string flavour = "RNN";
    if (vm.count("lstm"))
    {
        flavour = "LSTM";
        repnumber = 2;
    }
    else if (vm.count("gru"))
    {
        flavour = "GRU";
        repnumber = 1;
    }
    else if (vm.count("dglstm"))
    {
        flavour = "DGLSTM";
        repnumber = 2;
    }

    VOCAB_SIZE_SRC = sd.size();
    nparallel = vm["nparallel"].as<int>();
    mbsize = vm["mbsize"].as < int >();

    if (vm.count("beamsearchdecode"))
    {
        beam_search_decode = vm["beamsearchdecode"].as<int>();
    }

    if (vm.count("devel")) {
        cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
        unsigned min_dev_id = 0;
        devel = read_corpus(vm["devel"].as<string>(), min_dev_id, sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), vm.count("appendBOSEOS")> 0, vm.count("charlevel") > 0);
        devel_numturn2did = get_numturn2dialid(devel);
    }

    if (vm.count("testcorpus")) {
        cerr << "Reading test corpus from " << vm["testcorpus"].as<string>() << "...\n";
        unsigned min_dev_id = 0;
        testcorpus = read_corpus(vm["testcorpus"].as<string>(), min_dev_id, sd, kSRC_SOS, kSRC_EOS);
        test_numturn2did = get_numturn2dialid(testcorpus);
    }

    string fname;
    if (vm.count("parameters")) {
        fname = vm["parameters"].as<string>();
    }
    else {
        ostringstream os;
        os << "attentionwithintention"
            << '_' << LAYERS
            << '_' << HIDDEN_DIM
            << '_' << flavour
            << "-pid" << getpid() << ".params";
        fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    Trainer* sgd = nullptr;
    if (vm["trainer"].as<string>() == "momentum")
        sgd = new MomentumSGDTrainer(&model, 1e-6, vm["eta"].as<float>());
    if (vm["trainer"].as<string>() == "sgd")
        sgd = new SimpleSGDTrainer(&model, 1e-6, vm["eta"].as<float>());
    if (vm["trainer"].as<string>() == "adagrad")
        sgd = new AdagradTrainer(&model, 1e-6, vm["eta"].as<float>());
    sgd->clip_threshold = vm["clip"].as<float>();

    cerr << "%% Using " << flavour << " recurrent units" << endl;

    std::vector<unsigned> dims;
    dims.resize(4);
    if (!vm.count("hidden"))
        dims[ENCODER_LAYER] = HIDDEN_DIM;
    else
        dims[ENCODER_LAYER] = (unsigned)vm["hidden"].as<int>();
    dims[DECODER_LAYER] = dims[ENCODER_LAYER]; /// if not specified, encoder and decoder have the same dimension

    if (!vm.count("align"))
        dims[ALIGN_LAYER] = ALIGN_DIM;
    else
        dims[ALIGN_LAYER] = (unsigned)vm["align"].as<int>();
    if (!vm.count("intentiondim"))
        dims[INTENTION_LAYER] = HIDDEN_DIM;
    else
        dims[INTENTION_LAYER] = (unsigned)vm["intentiondim"].as<int>();
    rnn_t hred(model, LAYERS, VOCAB_SIZE_SRC, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, mem_slots, vm["scale"].as<float>());
    prt_model_info<rnn_t, TrainProc>(LAYERS, VOCAB_SIZE_SRC, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, mem_slots, vm["scale"].as<float>());

    if (vm.count("initialise"))
    {
        string fname = vm["initialise"].as<string>();
        ifstream in(fname, ifstream::in);
        if (in.is_open())
        {
            boost::archive::text_iarchive ia(in);
            ia >> model;
        }
    }

    ptrTrainer = new TrainProc();

    if (vm["pretrain"].as<float>() > 0)
    {
        ptrTrainer->supervised_pretrain(model, hred, training, devel, *sgd, fname, vm["pretrain"].as<float>(), 1);
        delete sgd;

        /// reopen sgd
        if (vm["trainer"].as<string>() == "momentum")
            sgd = new MomentumSGDTrainer(&model, 1e-6, vm["eta"].as<float>());
        if (vm["trainer"].as<string>() == "sgd")
            sgd = new SimpleSGDTrainer(&model, 1e-6, vm["eta"].as<float>());
        if (vm["trainer"].as<string>() == "adagrad")
            sgd = new AdagradTrainer(&model, 1e-6, vm["eta"].as<float>());
        sgd->clip_threshold = vm["clip"].as<float>();
    }

    if (vm.count("sampleresponses"))
    {
        cerr << "Reading sample corpus from " << vm["sampleresponses"].as<string>() << "...\n";
        unsigned min_dev_id = 0;
        training = read_corpus(vm["sampleresponses"].as<string>(), min_dev_id, sd, kSRC_SOS, kSRC_EOS);
        ptrTrainer->collect_sample_responses(hred, training);
    }
    if (vm.count("dialogue"))
    {
        if (vm.count("outputfile") == 0)
        {
            cerr << "missing recognition output file" << endl;
            abort();
        }
        ptrTrainer->dialogue(model, hred, vm["outputfile"].as<string>(), sd);
    }
    if (vm.count("nparallel") && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        ptrTrainer->batch_train(model, hred, training, devel, *sgd, fname, vm["epochs"].as<int>(), vm["nparallel"].as<int>());
    }
    else if (!vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        ptrTrainer->train(model, hred, training, devel, *sgd, fname, vm["epochs"].as<int>(), min_diag_id, vm.count("charlevel") > 0, vm.count("nosplitdialogue"));
    }
    else if (vm.count("testcorpus"))
    {
        if (vm.count("outputfile") == 0)
        {
            cerr << "missing recognition output file" << endl;
            abort();
        }
        ptrTrainer->test(model, hred, testcorpus, vm["outputfile"].as<string>(), sd, test_numturn2did, vm["scoreembeddingfn"].as<string>());
    }

    delete sgd;
    delete ptrTrainer;

    return EXIT_SUCCESS;
}

#endif
