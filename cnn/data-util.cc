#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/data-util.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace cnn;
using namespace std;

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& randstt)
{
    int nCorpus = raw.size();
    long nutt = raw[0]->size();

    vector<vector<Expression>> ret;
    random_device rd;

    vector<vector<cnn::real>> tgt;
    assert(randstt.size() == nutt);

    int feat_dim = (*raw[0])[0][0].size();
    for (auto cc : raw)
    {
        vector<vector<cnn::real>> obs;
        for (size_t k = 0; k < mbsize; k++)
        {
            obs.push_back(vector<cnn::real>(feat_dim * nutt, 0.0));
            for (size_t u = 0; u < nutt; u++)
            {
                size_t nsamples = (*cc)[u].size();
                if (k + randstt[u] >= nsamples)
                    break;

                size_t stt = u * feat_dim;

                /// random starting position randstt
                vector<cnn::real>::iterator pobs = obs[k].begin();
                vector<cnn::real>::iterator pfrm = (*cc)[u][k + randstt[u]].begin();
                copy(pfrm, pfrm + feat_dim, pobs + stt);
            }
        }

        vector<Expression> vx(mbsize);
        for (unsigned i = 0; i < mbsize; ++i)
        {
            vx[i] = input(cg, { (long)feat_dim, nutt }, &obs[i]);
            cg.incremental_forward();
        }
        ret.push_back(vx); 
    }

    return ret;
}

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs_uttfirst(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& randstt)
{
    int nCorpus = raw.size();
    long nutt = raw[0]->size();

    vector<vector<Expression>> ret;
    random_device rd;

    assert(randstt.size() == nutt);
    vector<vector<cnn::real>> tgt;
    int feat_dim = (*raw[0])[0][0].size();

    for (auto cc : raw)
    {
        vector<vector<cnn::real>> obs;
        for (size_t u = 0; u < nutt; u++)
        {
            obs.push_back(vector<cnn::real>(feat_dim * mbsize, 0.0));
            vector<cnn::real>::iterator pobs = obs[u].begin();

            for (size_t k = 0; k < mbsize; k++)
            {
                size_t nsamples = (*cc)[u].size();
                if (k + randstt[u] >= nsamples)
                    break;

                size_t stt = k * feat_dim;

                /// random starting position randstt
                vector<cnn::real>::iterator pfrm = (*cc)[u][k + randstt[u]].begin();
                copy(pfrm, pfrm + feat_dim, pobs + stt);
            }
        }

        vector<Expression> vx(nutt);
        for (unsigned i = 0; i < nutt; ++i)
        {
            vx[i] = input(cg, { (long)feat_dim, (long)mbsize}, &obs[i]);
            cg.incremental_forward();
        }
        ret.push_back(vx);
    }

    return ret;
}

/** 
extract from a dialogue corpus, a set of dialogues with the same number of turns
@corp : dialogue corpus
@nbr_dialogues : expected number of dialogues to extract
@stt_dialogue_id : starting dialogue id

return a vector of dialogues in selected, also the starting dialogue id is increased by one.
Notice that a dialogue might be used in multiple times

selected [ turn 0 : <query_00, answer_00> <query_10, answer_10>]
         [ turn 1 : <query_01, answer_01> <query_11, answer_11>]
*/
int get_same_length_dialogues(Corpus corp, size_t nbr_dialogues, size_t &stt_dialgoue_id, vector<bool>& used, vector<Dialogue>& selected)
{
    int nutt = 0;
    if (stt_dialgoue_id >= corp.size())
        return nutt;

    while (stt_dialgoue_id < corp.size() && used[stt_dialgoue_id])
        stt_dialgoue_id++;

    if (stt_dialgoue_id >= corp.size())
        return nutt;

    size_t d_turns = corp[stt_dialgoue_id].size();
    Dialogue first_turn;
    size_t i_turn = 0;

    selected.clear();
    selected.resize(d_turns);
    for (auto p : corp[stt_dialgoue_id])
    {
        selected[i_turn].push_back(corp[stt_dialgoue_id][i_turn]);
        i_turn++;
    }
    used[stt_dialgoue_id] = true;
    nutt++;

    for (size_t iss = stt_dialgoue_id+1; iss < corp.size(); iss++)
    {
        if (corp[iss].size() == d_turns && used[iss] == false){
            i_turn = 0;
            for (auto p : corp[iss])
            {
                selected[i_turn].push_back(corp[iss][i_turn]);
                i_turn++;
            }

            used[iss] = true;
            nutt++;
            if (selected[0].size() == nbr_dialogues)
                break;
        }
    }

    stt_dialgoue_id++;

    return nutt;
}

int MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td) 
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::vector<int>* v = s;
    std::string diagid, turnid;

    if (line.length() == 0)
        return -1;

    in >> diagid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting diagid" << endl;
        abort();
    }

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting turn id" << endl;
        abort();
    }

    while (in) {
        in >> word;
        if (!in) break;
        if (word == sep) { d = td; v = t; continue; }
        v->push_back(d->Convert(word));
    }
    int res;
    stringstream(diagid) >> res;
    return res;
}

Corpus read_corpus(const string &filename, unsigned& min_diag_id, Dict& sd, int kSRC_SOS, int kSRC_EOS)
{
    ifstream in(filename);
    assert(in);
    Corpus corpus;
    Dialogue diag;
    int prv_diagid = -1;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
    min_diag_id = 99999;
    while (getline(in, line)) {
        ++lc;
        Sentence source, target;
        int diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd);
        if (diagid == -1)
            break;
        if (diagid < min_diag_id)
            min_diag_id = diagid;
        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }
        diag.push_back(SentencePair(source, target));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }

    if (diag.size() > 0)
        corpus.push_back(diag);
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << sd.size() << " types\n";
    return corpus;
}

/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
Expression shuffle_data(Expression src, size_t nutt, size_t feat_dim, size_t slen)
{
    Expression i_src = reshape(src, {(long) (nutt * slen * feat_dim)});

    int stride = nutt * feat_dim;
    vector<Expression> i_all_spk;
    for (size_t k = 0; k < nutt; k++)
    {
        vector<Expression> i_each_spk;
        for (size_t t = 0; t < slen; t++)
        {
            long stt = k * feat_dim;
            long stp = (k + 1)*feat_dim;
            stt += (t * stride);
            stp += (t * stride);
            Expression i_pick = pickrange(i_src, stt, stp);
            i_each_spk.push_back(i_pick);
        }
        i_all_spk.push_back(concatenate_cols(i_each_spk));
    }
    return concatenate_cols(i_all_spk);
}