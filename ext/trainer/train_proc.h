#ifndef _TRAIN_PROC_H
#define _TRAIN_PROC_H

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/dnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/cnn-helper.h"
#include "ext/dialogue/attention_with_intention.h"
#include "ext/lda/lda.h"
#include "ext/ngram/ngram.h"
#include "cnn/data-util.h"
#include "cnn/grad-check.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>
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
unsigned VOCAB_SIZE_TGT = 0;
long nparallel = -1;
long mbsize = -1;
size_t g_train_on_turns = 1; 

cnn::Dict sd;
cnn::Dict td;

int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
int verbose;
int beam_search_decode;
cnn::real lambda = 1e-6;
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
    TrainProcess(){
    }

    void prt_model_info(size_t LAYERS, size_t VOCAB_SIZE_SRC, const vector<unsigned>& dims, size_t nreplicate, size_t decoder_additiona_input_to, size_t mem_slots, cnn::real scale);

    void batch_train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real& largest_cost, bool do_segmental_training, bool update_sgd);
    void supervised_pretrain(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, cnn::real target_ppl, int min_diag_id,
        bool bcharlevel = false, bool nosplitdialogue = false);
    void train(Model &model, Proc &am, Corpus &training, Corpus &devel,
        Trainer &sgd, string out_file, int max_epochs, 
        bool bcharlevel = false, bool nosplitdialogue = false);
    void train(Model &model, Proc &am, TupleCorpus &training, Trainer &sgd, string out_file, int max_epochs);
    void REINFORCEtrain(Model &model, Proc &am, Proc &am_agent_mirrow, Corpus &training, Corpus &devel, Trainer &sgd, string out_file, Dict & td, int max_epochs, int nparallel, cnn::real& largest_cost, cnn::real reward_baseline = 0.0, cnn::real threshold_prob_for_sampling = 1.0);
    void split_data_batch_train(string train_filename, Model &model, Proc &am, Corpus &devel, Trainer &sgd, string out_file, int max_epochs, int nparallel, int epochsize, bool do_segmental_training);
    void test(Model &model, Proc &am, Corpus &devel, string out_file, Dict & td, NumTurn2DialogId& test_corpusinfo, const string& score_embedding_fn = "");
    void test(Model &model, Proc &am, Corpus &devel, string out_file, Dict & sd);
    void test(Model &model, Proc &am, TupleCorpus &devel, string out_file, Dict & sd, Dict & td);
    void dialogue(Model &model, Proc &am, string out_file, Dict & td);

    void collect_sample_responses(Proc& am, Corpus &training);

    void nosegmental_forward_backward(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt,
        cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr);
    void segmental_forward_backward(Model &model, Proc &am, PDialogue &v_v_dialogues, int nutt,
        cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr);
    void REINFORCE_nosegmental_forward_backward(Model &model, Proc &am, Proc &am_mirrow, PDialogue &v_v_dialogues, int nutt,
        cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, Trainer* sgd, Dict& sd, cnn::real reward_baseline = 0.0, cnn::real threshold_prob_for_sampling = 1.0,
        bool update_model = true);

    cnn::real smoothed_ppl(cnn::real curPPL);
    void reset_smoothed_ppl(){
        ppl_hist.clear();
    }

public:
    /// for LDA
    void lda_train(variables_map vm, const Corpus &training, const Corpus &test, Dict& sd);
    void lda_test(variables_map vm, const Corpus& test, Dict& sd);

public:
    /// for ngram
    void ngram_train(variables_map vm, const Corpus& test, Dict& sd);
    void ngram_clustering(variables_map vm, const Corpus& test, Dict& sd);

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
    cnn::real dloss = 0;
    cnn::real dchars_s  = 0;
    cnn::real dchars_t = 0;

    ofstream of(out_file);

    unsigned si = devel.size(); /// number of dialgoues in training

    Timer iteration("completed in");

    /// report BLEU score
    test(model, am, devel, out_file + "bleu", sd);

    cnn::real ddloss = 0;
    cnn::real ddchars_s = 0;
    cnn::real ddchars_t = 0;

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

        cnn::real emb_loss = 0;
        cnn::real emb_chars_s = 0;
        cnn::real emb_chars_t = 0;
        cnn::real turns = 0;
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

/** warning, the test function use the true past response as the context, when measure bleu score
• So the BLEU score is artificially high
• However, because the use input is conditioned on the past response. If using the true decoder response as the past context, the user input cannot be from the corpus.
• Therefore, it is reasonable to use the true past response as context when evaluating the model.
*/
template <class AM_t>
void TrainProcess<AM_t>::test(Model &model, AM_t &am, Corpus &devel, string out_file, Dict & sd)
{
    unsigned lines = 0;
    cnn::real dloss = 0;
    cnn::real dchars_s = 0;
    cnn::real dchars_t = 0;

    BleuMetric bleuScore; 
    bleuScore.Initialize();

    ofstream of(out_file);

    unsigned si = devel.size(); /// number of dialgoues in training

    Timer iteration("completed in");
    cnn::real ddloss = 0;
    cnn::real ddchars_s = 0;
    cnn::real ddchars_t = 0;


    for (auto diag : devel){

        SentencePair prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        ComputationGraph cg;
        vector<int> res;
        for (auto spair : diag){

            SentencePair turn = spair;
            vector<string> sref, srec;

            if (turn_id == 0)
                res = am.decode(turn.first, cg, sd);
            else
                res = am.decode(prv_turn.second, turn.first, cg, sd);

            if (turn.first.size() > 0)
            {
                cout << "source: ";
                for (auto p : turn.first){
                    cout << sd.Convert(p) << " ";
                }
                cout << endl;
            }

            if (turn.second.size() > 0)
            {
                cout << "ref response: ";
                for (auto p : turn.second){
                    cout << sd.Convert(p) << " ";
                    sref.push_back(sd.Convert(p));
                }
                cout << endl;
            }

            if (res.size() > 0)
            {
                cout << "res response: ";
                for (auto p : res){
                    cout << sd.Convert(p) << " ";
                    srec.push_back(sd.Convert(p));
                }
                cout << endl;
            }


            bleuScore.AccumulateScore(sref, srec);

            turn_id++;
            prv_turn = turn;
        }
    }

    string sBleuScore = bleuScore.GetScore(); 
    cout << "BLEU (4) score = " << sBleuScore << endl;
    of << sBleuScore << endl;

    of.close();
}

/**
Test on the tuple corpus 
output recognition results for each test
not using perplexity to report progresses
*/
template <class AM_t>
void TrainProcess<AM_t>::test(Model &model, AM_t &am, TupleCorpus &devel, string out_file, Dict & sd, Dict & td)
{
    unsigned lines = 0;
    cnn::real dloss = 0;
    cnn::real dchars_s = 0;
    cnn::real dchars_t = 0;

    ofstream of(out_file);

    unsigned si = devel.size(); /// number of dialgoues in training

    Timer iteration("completed in");
    cnn::real ddloss = 0;
    cnn::real ddchars_s = 0;
    cnn::real ddchars_t = 0;

    for (auto diag : devel){

        SentenceTuple prv_turn;
        size_t turn_id = 0;

        /// train on two segments of a dialogue
        ComputationGraph cg;
        vector<int> res;
        for (auto spair : diag){

            SentenceTuple turn = spair;

            if (turn_id == 0)
                res = am.decode_tuple(turn, cg, sd, td);
            else
                res = am.decode_tuple(prv_turn, turn, cg, sd, td);

            if (turn.first.size() > 0)
            {
                for (auto p : turn.first){
                    cout << sd.Convert(p) << " ";
                }
                cout << endl;
            }

            if (turn.last.size() > 0)
            {
                for (auto p : turn.last){
                    cout << sd.Convert(p) << " ";
                }
                cout << endl;
            }

            if (res.size() > 0)
            {
                for (auto p : res){
                    cout << td.Convert(p) << " ";
                }
                cout << endl;
            }

            turn_id++;
            prv_turn = turn;
        } 
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

/**
inspired by the following two papers
Sequence level training with recurrent neural networks http://arxiv.org/pdf/1511.06732v3.pdf
Minimum risk training for neural machine translation http://arxiv.org/abs/1512.02433

use decoded responses as targets. start this process from the last turn, and then gradually move to earlier turns.
this is also for implementation convenience.

/// initially alwasy use the xent, later on, with probability p, use the decoded response as target, but weight it
/// with a reward from BLEU
/// this probability is increased from 0 to 1.0.
/// two avoid different scaling, should apply decoding to all incoming sentences or otherwise, all use xent training

/// with probability p, decode an input
vector<int> response = s2tmodel_sim.decode(insent, cg);
/// evaluate the response to get BLEU score

/// subtract the BLEU score with a baseline number

/// the scalar is the reward signal

/// the target responses: some utterances are with true responses and the others are with decoded responses
*/
template <class AM_t>
void TrainProcess<AM_t>::REINFORCE_nosegmental_forward_backward(Model &model, AM_t &am, AM_t &am_mirrow, PDialogue &v_v_dialogues, int nutt,
    cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, Trainer* sgd, Dict& sd, cnn::real reward_baseline, cnn::real threshold_prob_for_sampling, bool update_model)
{
    size_t turn_id = 0;
    size_t i_turns = 0;
    PTurn prv_turn, new_turn, new_prv_turn;
    BleuMetric bleuScore;
    bleuScore.Initialize();

    bool do_sampling = false; 
    cnn::real rng_value = rand() / (RAND_MAX + 0.0); 
    if (rng_value >= threshold_prob_for_sampling)
    {
        do_sampling = true; 
    }

    ComputationGraph cg;

    am.reset();
    am_mirrow.reset();

    /// train on two segments of a dialogue
    vector<Sentence> res;
    vector<Expression> v_errs; /// the errors to be minimized
    vector<cnn::real> v_bleu_score;
    vector<Expression> i_err;

    for (auto &turn : v_v_dialogues)
    {
        if (do_sampling)
        {
            vector<string> sref, srec;
            vector<Sentence> v_input, v_prv_response;

            v_bleu_score.clear();

            for (auto& p : turn)
            {
                v_input.push_back(p.first);
            }
            for (auto&p : prv_turn)
            {
                v_prv_response.push_back(p.second);
            }

            if (turn_id == 0)
            {
                res = am_mirrow.batch_decode(v_input, cg, sd);
            }
            else
            {
                res = am_mirrow.batch_decode(v_prv_response, v_input, cg, sd);
            }

            size_t k = 0;
            for (auto &p : res)
            {

                sref.clear();
                if (verbose) cout << "ref response: ";
                for (auto p : turn[k].second){
                    if (verbose) cout << sd.Convert(p) << " ";
                    sref.push_back(sd.Convert(p));
                }
                if (verbose) cout << endl;

                srec.clear();
                if (verbose) cout << "res response: ";
                for (auto p : res[k]){
                    if (verbose) cout << sd.Convert(p) << " ";
                    srec.push_back(sd.Convert(p));
                }
                if (verbose) cout << endl;

                cnn::real score = bleuScore.GetSentenceScore(sref, srec);
                v_bleu_score.push_back(score);

                k++;
            }

            new_turn = turn;
            for (size_t k = 0; k < nutt; k++)
            {
                new_turn[k].second = res[k];
            }

            /// get errors from the decoded results
            if (turn_id == 0)
            {
                i_err = am.build_graph(new_turn, cg);
            }
            else
            {
                i_err = am.build_graph(new_prv_turn, new_turn, cg);
            }
        }
        else{
            /// get errors from the true reference
            if (turn_id == 0)
            {
                i_err = am.build_graph(turn, cg);
            }
            else
            {
                i_err = am.build_graph(prv_turn, turn, cg);
            }
        }

        if (do_sampling)
        {
            for (size_t k = 0; k < nutt; k++)
            {
                Expression t_err = i_err[k];
                v_errs.push_back(t_err * (v_bleu_score[k] - reward_baseline));  /// multiply with reward
            }
        }
        else
        {
            for (auto &p : i_err)
                v_errs.push_back(p);
        }

        cg.incremental_forward();

        prv_turn = turn;
        new_prv_turn = new_turn;
        turn_id++;
        i_turns++;
    }

    Expression i_total_err = sum(v_errs);
    dloss += as_scalar(cg.get_value(i_total_err));

    dchars_s += am.swords;
    dchars_t += am.twords;

    if (sgd != nullptr && update_model)
    {
        cg.backward();
        sgd->update(am.twords);
    }
}

template <class AM_t>
void TrainProcess<AM_t>::nosegmental_forward_backward(Model &model, AM_t &am, PDialogue &v_v_dialogues, int nutt,
    cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr)
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

template <class AM_t>
void TrainProcess<AM_t>::segmental_forward_backward(Model &model, AM_t &am, PDialogue &v_v_dialogues, int nutt,
    cnn::real &dloss, cnn::real & dchars_s, cnn::real & dchars_t, bool resetmodel = false, int init_turn_id = 0, Trainer* sgd = nullptr)
{
    size_t turn_id = init_turn_id;
    size_t i_turns = 0;
    PTurn prv_turn;

    for (auto turn : v_v_dialogues)
    {
        ComputationGraph cg;
        if (resetmodel)
        {
            am.reset();
        }

        if (turn_id == 0)
        {
            am.build_graph(turn, cg);
        }
        else
        {
            am.build_graph(prv_turn, turn, cg);
        }

        cg.incremental_forward();
        if (sgd != nullptr)
        {
            cg.backward();
            sgd->update(am.twords);
        }

        dloss += as_scalar(cg.get_value(am.s2txent.i));

        dchars_s += am.swords;
        dchars_t += am.twords;

        prv_turn = turn;
        turn_id++;
        i_turns++;
    }

}

/**
Train with REINFORCE algorithm
*/
template <class AM_t>
void TrainProcess<AM_t>::REINFORCEtrain(Model &model, AM_t &am, AM_t &am_agent_mirrow, Corpus &training, Corpus &devel, Trainer &sgd, string out_file, Dict & td, int max_epochs, int nparallel, cnn::real& largest_cost, cnn::real reward_baseline, cnn::real threshold_prob_for_sampling)
{
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    threshold_prob_for_sampling = min<cnn::real>(1.0, max<cnn::real>(0.0, threshold_prob_for_sampling)); /// normalize to [0.0, 1.0]

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
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;
        cnn::real dchars_tt = 0;

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

            Dialogue prv_turn;
            
            PDialogue v_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
            vector<int> i_sel_idx = get_same_length_dialogues(training, nparallel, i_stt_diag_id, v_selected, v_dialogues, training_numturn2did);
            size_t nutt = i_sel_idx.size();

            REINFORCE_nosegmental_forward_backward(model, am, am_agent_mirrow, v_dialogues, nutt, dloss, dchars_s, dchars_t, &sgd, td, reward_baseline, threshold_prob_for_sampling);
            si += nutt;
            lines += nutt;
            iter += nutt;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

        // show score on dev data?
        report++;
        if (floor(sgd.epoch) != prv_epoch || report % dev_every_i_reports == 0 || fmod(lines, (cnn::real)training.size()) == 0.0) {
            cnn::real ddloss = 0;
            cnn::real ddchars_s = 0;
            cnn::real ddchars_t = 0;

            vector<bool> vd_selected(devel.size(), false);  /// track if a dialgoue is used
            size_t id_stt_diag_id = 0;
            PDialogue vd_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers
            vector<int> id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
            size_t ndutt = id_sel_idx.size();

            while (ndutt > 0)
            {
                /// the cost is -(r - r_baseline) * log P
                /// for small P, but with large r, the cost is high, so to reduce it, it generates large gradient as this event corresponds to low probability but high reward
                REINFORCE_nosegmental_forward_backward(model, am, am_agent_mirrow, vd_dialogues, ndutt, ddloss, ddchars_s, ddchars_t, nullptr, td, reward_baseline, 0.0, false);
                
                id_sel_idx = get_same_length_dialogues(devel, NBR_DEV_PARALLEL_UTTS, id_stt_diag_id, vd_selected, vd_dialogues, devel_numturn2did);
                ndutt = id_sel_idx.size();
            }
            ddloss = smoothed_ppl(ddloss);
            if (ddloss < largest_cost) {
                largest_cost = ddloss;
                ofstream out(out_file, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else{
                sgd.eta0 *= 0.5; /// reduce learning rate
                sgd.eta *= 0.5; /// reduce learning rate
            }
            cerr << "\n***DEV [epoch=" << (lines / (cnn::real)training.size()) << "] cost = " << (ddloss / ddchars_t) << " approximate ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);
    }
}


/* the following does mutiple sentences per minibatch
but I comment it out 
*/
template <class AM_t>
void TrainProcess<AM_t>::batch_train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, int nparallel, cnn::real &best, bool segmental_training,
    bool sgd_update_epochs)
{
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 1000;
    unsigned si = training.size(); /// number of dialgoues in training
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    reset_smoothed_ppl();

    int prv_epoch = -1;
    vector<bool> v_selected(training.size(), false);  /// track if a dialgoue is used
    size_t i_stt_diag_id = 0;

    while (sgd.epoch < max_epochs) {
        Timer iteration("completed in");
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;
        cnn::real dchars_tt = 0;

        PDialogue v_dialogues;  // dialogues are orgnaized in each turn, in each turn, there are parallel data from all speakers

        for (unsigned iter = 0; iter < report_every_i;) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else if (sgd_update_epochs){
                    sgd.update_epoch();
                    lines -= training.size();
                }
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

            Dialogue prv_turn;
            size_t turn_id = 0;
            vector<int> i_sel_idx = get_same_length_dialogues(training, nparallel, i_stt_diag_id, v_selected, v_dialogues, training_numturn2did);
            size_t nutt = i_sel_idx.size();
            if (nutt == 0)
                break;

            if (verbose)
            {
                cerr << "selected " << nutt << " :  ";
                for (auto p : i_sel_idx)
                    cerr << p << " ";
                cerr << endl;
            }

            if (segmental_training)
                segmental_forward_backward(model, am, v_dialogues, nutt, dloss, dchars_s, dchars_t, false, 0, &sgd);
            else
                nosegmental_forward_backward(model, am, v_dialogues, nutt, dloss, dchars_s, dchars_t, true, 0, &sgd);

            si += nutt;
            lines += nutt;
            iter += nutt;
        }

        sgd.status();
        iteration.WordsPerSecond(dchars_t + dchars_s);
        cerr << "\n***Train " << (lines / (cnn::real)training.size()) * 100 << " %100 of epoch[" << sgd.epoch << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';


        vector<SentencePair> vs;
        for (auto&p : v_dialogues)
            vs.push_back(p[0]);
        vector<SentencePair> vres;
        am.respond(vs, vres, sd);
        
        // show score on dev data?
        report++;

        if (devel.size() > 0 && (floor(sgd.epoch) != prv_epoch || report % dev_every_i_reports == 0 || fmod(lines, (cnn::real)training.size()) == 0.0)) {
            cnn::real ddloss = 0;
            cnn::real ddchars_s = 0;
            cnn::real ddchars_t = 0;

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
            cerr << "\n***DEV [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);

        if (sgd_update_epochs == false)
        {
            /// because there is no update on sgd epoch, this loop can run forever. 
            /// so just run one iteration and quit
            break;
        }
    }
}

/**
@bcharlevel : true if character output; default false.
*/
template <class AM_t>
void TrainProcess<AM_t>::train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, int max_epochs, bool bcharlevel = false, bool nosplitdialogue = false)
{
    cnn::real best = 9e+99;
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
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;
        cnn::real dchars_tt = 0;

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
            auto& spair = training[order[si % order.size()]];

            if (verbose)
                cerr << "diag = " << order[si % order.size()] << endl;

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
                        cnn::real tcxtent = as_scalar(cg.get_value(am.s2txent));
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
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';


        // show score on dev data?
        report++;
        if (floor(sgd.epoch) != prv_epoch || report % dev_every_i_reports == 0 || fmod(lines, (cnn::real)training.size()) == 0.0) {
            cnn::real ddloss = 0;
            cnn::real ddchars_s = 0;
            cnn::real ddchars_t = 0;

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
            cerr << "\n***DEV [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (ddloss / ddchars_t) << " ppl=" << exp(ddloss / ddchars_t) << ' ';
        }

        prv_epoch = floor(sgd.epoch);
    }
}

/**
Training process on tuple corpus
*/
template <class AM_t>
void TrainProcess<AM_t>::train(Model &model, AM_t &am, TupleCorpus &training, Trainer &sgd, string out_file, int max_epochs)
{
    cnn::real best = 9e+99;
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
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;
        cnn::real dchars_tt = 0;

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
            auto& spair = training[order[si % order.size()]];

            if (verbose)
                cerr << "diag = " << order[si % order.size()] << endl;

            /// find portion to train
            bool b_trained = false;
            // see random number distributions
            auto rng = std::bind(std::uniform_int_distribution<int>(0, spair.size() - 1), *rndeng);
            int i_turn_to_train = rng();

            vector<SentenceTuple> prv_turn;
            size_t turn_id = 0;

            size_t i_init_turn = 0;

            /// train on two segments of a dialogue
            ComputationGraph cg;
            size_t t = 0;
            do{
                if (i_init_turn > 0)
                    am.assign_cxt(cg, 1);

                SentenceTuple turn = spair[t];
                vector<SentenceTuple> i_turn(1, turn);
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

                t++;
                prv_turn = i_turn;
            } while (t < spair.size());

            dloss += as_scalar(cg.get_value(am.s2txent.i));

            dchars_s += am.swords;
            dchars_t += am.twords;

            //            CheckGrad(model, cg);
            cg.backward();
            sgd.update(am.twords);

            if (verbose)
                cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';
            ++si;
            lines++;
        }
        sgd.status();
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)training.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

        if (fmod(lines , (cnn::real)training.size()) == 0)
        {
            cnn::real i_ppl = smoothed_ppl(exp(dloss / dchars_t));
            if (best > i_ppl)
            {
                best = i_ppl;

                ofstream out(out_file, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else
            {
                sgd.eta0 *= 0.5;
                sgd.eta *= 0.5;
            }
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
    size_t k = 0;
    for (auto p : ppl_hist)
    {
        finPPL += p;
        k++;
    }
    return finPPL/k;
}

/**
overly pre-train models on small subset of the data 
*/
template <class AM_t>
void TrainProcess<AM_t>::supervised_pretrain(Model &model, AM_t &am, Corpus &training, Corpus &devel,
    Trainer &sgd, string out_file, cnn::real target_ppl, int min_diag_id,
    bool bcharlevel = false, bool nosplitdialogue = false)
{
    cnn::real best = 9e+99;
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
        cnn::real dloss = 0;
        cnn::real dchars_s = 0;
        cnn::real dchars_t = 0;
        cnn::real dchars_tt = 0;

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
                        cnn::real tcxtent = as_scalar(cg.get_value(am.s2txent));
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
        cerr << "\n***Train [epoch=" << (lines / (cnn::real)order.size()) << "] E = " << (dloss / dchars_t) << " ppl=" << exp(dloss / dchars_t) << ' ';

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

/** 
since the tool loads data into memory and that can cause memory exhaustion, this function do sampling of data for each epoch.
*/
template <class AM_t>
void TrainProcess<AM_t>::split_data_batch_train(string train_filename, Model &model, AM_t &am, Corpus &devel, 
    Trainer &sgd, string out_file, 
    int max_epochs, int nparallel, int epochsize, bool segmental_training)
{
    // a mirrow of the agent to generate decoding results so that their results can be evaluated
    // this is not efficient implementation, better way is to share model parameters
    cnn::real largest_cost = 9e+99;

    ifstream ifs(train_filename);
    int trial = 0;
    while(sgd.epoch < max_epochs)
    {
        cerr << "Reading training data from " << train_filename << "...\n";
        Corpus training = read_corpus(ifs, sd, kSRC_SOS, kSRC_EOS, epochsize);
        training_numturn2did = get_numturn2dialid(training);

        if (ifs.eof() || training.size() == 0)
        {
            ifs.close();
            ifs.open(train_filename); 
            
            if (training.size() == 0)
            {
                continue;
            }
            ofstream out(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), ofstream::out);
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();

            sgd.update_epoch();
        }

        batch_train(model, am, training, devel, sgd, out_file, 1, nparallel, largest_cost, segmental_training, false);

        if (fmod(trial , 50) == 0)
        {
            ofstream out(out_file + ".i" + boost::lexical_cast<string>(sgd.epoch), ofstream::out);
            boost::archive::text_oarchive oa(out);
            oa << model;
            out.close();
        }
        trial++;
    }
    ifs.close();
}

/**
@bcharlevel : true if character output; default false.
*/
template <class AM_t>
void TrainProcess<AM_t>::lda_train(variables_map vm, const Corpus &training, const Corpus& test, Dict& sd)
{
    ldaModel * pLda = new ldaModel(training, test);

    pLda->init(vm);

    pLda->read_data(training, sd, test);

    pLda->train();
    pLda->save_ldaModel_topWords(vm["lda-model"].as<string>() + ".topic.words", sd);

    pLda->load_ldaModel(-1);
    pLda->test(sd);

    delete[] pLda;
}

/**
@bcharlevel : true if character output; default false.
*/
template <class AM_t>
void TrainProcess<AM_t>::lda_test(variables_map vm, const Corpus& test, Dict& sd)
{
    Corpus empty;
    ldaModel * pLda = new ldaModel(empty, test);

    pLda->init(vm);
    pLda->load_ldaModel(vm["lda-final-model"].as<string>());

    pLda->read_data(empty, sd, test);

    pLda->test(sd);

    delete[] pLda;
}

/**
train n-gram model
*/
template <class AM_t>
void TrainProcess<AM_t>::ngram_train(variables_map vm, const Corpus& test, Dict& sd)
{
    Corpus empty;
    nGram pnGram= nGram();
    pnGram.Initialize(vm);

    for (auto & t : test)
    {
        for (auto & s : t)
        {
            pnGram.UpdateNgramCounts(s.second, 0, sd);
            pnGram.UpdateNgramCounts(s.second, 1, sd);
        }
    }

    pnGram.ComputeNgramModel();

    pnGram.SaveModel(); 

}

/**
cluster using n-gram model
random shuffle training data and use the first half to train ngram model
after several iterations, which have their log-likelihood reported, the ngram model assign a class id for each sentence
*/
template <class AM_t>
void TrainProcess<AM_t>::ngram_clustering(variables_map vm, const Corpus& test, Dict& sd)
{
    Corpus empty;
    int ncls = vm["ngram-num-clusters"].as<int>();
    vector<nGram> pnGram(ncls); 
    for (auto& p: pnGram)
        p.Initialize(vm);

    /// flatten corpus
    Sentences user, response;
    flatten_corpus(test, user, response);

    for (int iter = 0; iter < 10; iter++)
    {
        shuffle(response.begin(), response.end(), std::default_random_engine(iter));
        if (iter == 0)
        {
            for (int i = 0; i < ncls; i++)
            {
                pnGram[i].LoadModel(".m" + boost::lexical_cast<string>(i));
            }

            for (int i = 0; i < response.size(); i++)
            {
                int cls = rand0n_uniform(ncls - 1);
                pnGram[cls].UpdateNgramCounts(response[i], 0, sd);
            }
#pragma omp parallel for
            for (int i = 0; i < ncls; i++)
            {
                pnGram[i].ComputeNgramModel();
                pnGram[i].SaveModel(".m" + boost::lexical_cast<string>(i));
            }
        }
        else
        {
            /// use half the data to update centroids

            /// reassign data to closest cluster
            vector<Sentences> current_assignment(ncls);
            double totallk = 0;
            for (int i = 0; i < response.size() / 2; i++)
            {
                vector<cnn::real> llk(ncls);
                for (int i = 0; i < ncls; i++)
                    llk[i] = pnGram[i].GetSentenceLL(response[i]);

                cnn::real largest = llk[0];
                int iarg = 0;
                for (int i = 1; i < ncls; i++)
                {
                    if (llk[i] > largest)
                    {
                        largest = llk[i];
                        iarg = i;
                    }
                }
                current_assignment[iarg].push_back(response[i]);
                totallk += largest / response[i].size();
            }
            totallk /= response.size();

            cout << "loglikelihood at iteration " << iter << " is " << totallk << endl;

            ///update cluster
#pragma omp parallel for
            for (int i = 0; i < ncls; i++)
                pnGram[i].Clear();

            for (int i = 0; i < current_assignment.size(); i++)
            {
                for (auto & p : current_assignment[i])
                {
                    pnGram[i].UpdateNgramCounts(p, 0, sd);
                    pnGram[i].UpdateNgramCounts(p, 1, sd);
                }
            }

#pragma omp parallel for
            for (int i = 0; i < ncls; i++)
            {
                pnGram[i].ComputeNgramModel();
                pnGram[i].SaveModel(".m" + boost::lexical_cast<string>(i));
            }
        }
    }

    /// do classification now
    ofstream ofs;
    if (vm.count("outputfile") > 0)
        ofs.open(vm["outputfile"].as<string>());
    long did = 0;
    for (auto& t : test)
    {
        int tid = 0;
        for (auto& s : t)
        {
            vector<cnn::real> llk(ncls);
            for (int i = 0; i < ncls; i++)
                llk[i] = pnGram[i].GetSentenceLL(s.second);

            cnn::real largest = llk[0];
            int iarg = 0;
            for (int i = 1; i < ncls; i++)
            {
                if (llk[i] > largest)
                {
                    largest = llk[i];
                    iarg = i;
                }
            }

            string userstr;
            for (auto& p : s.first)
                userstr = userstr + " " + sd.Convert(p);
            string responsestr;
            for (auto& p : s.second)
                responsestr = responsestr + " " + sd.Convert(p);

            string ostr = boost::lexical_cast<string>(did)+ " ||| " + boost::lexical_cast<string>(tid)+ " ||| " + userstr + " ||| " + responsestr;
            ostr = ostr + " ||| " + boost::lexical_cast<string>(iarg);
            if (ofs.is_open())
            {
                ofs << ostr << endl;
            }
            else
                cout << ostr << endl;

            tid++;
        }
        did++;
    }

    if (ofs.is_open())
        ofs.close();
}


#endif
