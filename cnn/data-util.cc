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
#include <algorithm>
#include <string>
#include <vector>
#include <tuple>
#include <functional>
#include <boost/system/config.hpp>
#include <boost/locale.hpp>
#include <boost/locale/encoding_utf.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <tuple>

using namespace cnn;
using namespace std;
using namespace boost::algorithm;
using boost::locale::conv::utf_to_utf;
using namespace boost::locale;

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& randstt)
{
    int nCorpus = raw.size();
    unsigned nutt = raw[0]->size();

    vector<vector<Expression>> ret;
    random_device rd;

    vector<vector<cnn::real>> tgt;
    assert(randstt.size() == nutt);

    unsigned feat_dim = (*raw[0])[0][0].size();
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
            vx[i] = input(cg, { feat_dim, nutt }, &obs[i]);
            cg.incremental_forward();
        }
        ret.push_back(vx); 
    }

    return ret;
}

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs_uttfirst(FCorpusPointers raw, unsigned mbsize, ComputationGraph& cg, const vector<size_t>& randstt)
{
    int nCorpus = raw.size();
    unsigned nutt= raw[0]->size();

    vector<vector<Expression>> ret;
    random_device rd;

    assert(randstt.size() == nutt);
    vector<vector<cnn::real>> tgt;
    unsigned feat_dim= (*raw[0])[0][0].size();

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
            vx[i] = input(cg, { feat_dim, mbsize}, &obs[i]);
            cg.incremental_forward();
        }
        ret.push_back(vx);
    }

    return ret;
}

/**
@param padding_to_the_back: true if padding is to the back of the input sequence; false if padding is to the front of the sequence
*/
PDialogue padding_with_eos(const PDialogue& v_diag, int padding_symbol, const std::vector<bool>& padding_to_the_back)
{
    PDialogue res;
    assert(padding_to_the_back.size() == 2 || padding_to_the_back.size() == 1);

    for (auto& t : v_diag)
    {
        int max_src = -1;
        int max_tgt = -1;
        for (auto &sp : t)
        {
            max_src = std::max<int>(sp.first.size(), max_src);  /// max source side length
            max_tgt = std::max<int>(sp.second.size(), max_tgt); /// max target side length
        }

        PTurn i_turn;
        for (int p = 0; p < t.size(); p++)
        {
            Sentence src(max_src, padding_symbol);
            Sentence tgt(max_tgt, padding_symbol);

            bool padding_back = padding_to_the_back[0];
            if (padding_back)
                std::copy_n(t[p].first.begin(), t[p].first.size(), src.begin());
            else
                std::copy_n(t[p].first.begin(), t[p].first.size(), src.end() - t[p].first.size());

            padding_back = (padding_to_the_back.size() == 2) ? padding_to_the_back[1] : padding_to_the_back[0];
            if (padding_back)
                std::copy_n(t[p].second.begin(), t[p].second.size(), tgt.begin());
            else
                std::copy_n(t[p].second.begin(), t[p].second.size(), tgt.end() - t[p].second.size());

            i_turn.push_back(make_pair(src, tgt));
        }
        res.push_back(i_turn);
    }

    return res;
}

/**
@param padding_to_the_back: true if padding is to the back of the input sequence; false if padding is to the front of the sequence
*/
Sentences padding_with_eos(const Sentences& v_sent, int padding_symbol, bool  padding_to_the_back)
{
    Sentences res;

    int max_src = -1;
    for (auto &sp : v_sent)
    {
        max_src = std::max<int>(sp.size(), max_src);  /// max source side length
    }

    for (auto& s : v_sent)
    {
        Sentence src(max_src, padding_symbol);

        if (padding_to_the_back)
            std::copy_n(s.begin(), s.size(), src.begin());
        else
            std::copy_n(s.begin(), s.size(), src.end() - s.size());
        res.push_back(src);
    }

    return res;
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
vector<int> get_same_length_dialogues(Corpus corp, int nbr_dialogues, size_t &min_nbr_turns, vector<bool>& used, PDialogue& selected, NumTurn2DialogId& info)
{
    int nutt = 0;
    vector<int> v_sel_idx;
    int nbr_turn = -1;

    for (auto n : info.vNumTurns)
    {
        for (auto k : info.mapNumTurn2DialogId[n])
        {
            if (used[k] == false)
            {
                nbr_turn = n;
            }
            if (nbr_turn != -1)
                break;
        }
        if (nbr_turn!= -1)
            break;
    }

    if (nbr_turn == -1)
        return v_sel_idx;

    selected.clear();
    selected.resize(nbr_turn);
    vector<int> vd = info.mapNumTurn2DialogId[nbr_turn];

    size_t nd = 0;
    for (auto k : vd)
    {
        if (used[k] == false && (nbr_dialogues < 0 ||(nd < nbr_dialogues && nbr_dialogues >= 0)))
        {
            size_t iturn = 0;
            for (auto p : corp[k])
            {
                selected[iturn].push_back(corp[k][iturn]);
                iturn++;
            }
            used[k] = true;
            v_sel_idx.push_back(k);
            nd++;
        }
        if (nbr_dialogues >= 0 && nd >= nbr_dialogues)
            break;
    }

    min_nbr_turns = nbr_turn;
    return v_sel_idx; 
}

std::wstring utf8_to_wstring(const std::string& str)
{
    return utf_to_utf<wchar_t>(str.c_str(), str.c_str() + str.size());
}

std::string wstring_to_utf8(const std::wstring& str)
{
    return utf_to_utf<char>(str.c_str(), str.c_str() + str.size());
}

/** flatten corous to the following
 vector<SentencePair> -> merge(perv_response, current_user) to a sentence
*/
void flatten_corpus(const Corpus& corpus, vector<Sentence>& sentences, vector<Sentence>& response)
{
    for (auto& p : corpus)
    {
        int iturn = 0;
        Sentence prv_response; 
        for (auto& d : p)
        {
            if (iturn == 0)
            {
                sentences.push_back(d.first);
            }
            else{
                Sentence this_doc = prv_response;
                this_doc.insert(this_doc.end(), d.first.begin(), d.first.end());
                sentences.push_back(this_doc);
            }

            response.push_back(d.second);
            prv_response = d.second;
            iturn++;
        }
    }
}

/** flatten corous to the following
vector<SentencePair> -> merge(perv_response, current_user) to a sentence
*/
void flatten_corpus(const CorpusWithClassId& corpus, vector<Sentence>& sentences, vector<SentenceWithId>& response)
{
    for (auto& p : corpus)
    {
        int iturn = 0;
        for (auto& d : p)
        {
            sentences.push_back(d.first);
            response.push_back(d.second);
            iturn++;
        }
    }
}

Corpus read_corpus(const string &filename, unsigned& min_diag_id, WDict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength, bool appendBSandES)
{
    wifstream in(filename);
    generator gen;
    locale loc  = gen("zh-CN.UTF-8");
    // Create all locales

    in.imbue(loc); 
    wstring line;

    Corpus corpus;
    Dialogue diag;
    int prv_diagid = -1;
    int lc = 0, stoks = 0, ttoks = 0;
    min_diag_id = 99999;
    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, target;
        int diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, appendBSandES, kSRC_SOS, kSRC_EOS);
        if (diagid == -1)
            continue;
        if (diagid < min_diag_id)
            min_diag_id = diagid;
        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }
        if (source.size() > maxSentLength)
        {
            source.resize(maxSentLength - 1);
            source.push_back(kSRC_EOS);
        }
        if (target.size() > maxSentLength)
        {
            target.resize(maxSentLength - 1);
            target.push_back(kSRC_EOS);
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

CorpusWithClassId read_corpus_with_classid(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS)
{
    ifstream in(filename);
    string line;

    CorpusWithClassId corpus;
    DialogueWithClassId diag;
    string prv_diagid = "-1";
    int lc = 0, stoks = 0, ttoks = 0;

    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            continue;
        ++lc;
        Sentence source, target, clsid;
        string diagid = MultiTurnsReadSentencePairWithClassId(line, &source, &sd, &target, &sd, &clsid, kSRC_SOS, kSRC_EOS);
        if (diagid.size() == 0)
            continue;

        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }

        diag.push_back(SentencePairAndClassId(make_pair(source, make_pair(target, clsid[0]))));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS)) {
            throw("Sentence in didn't start or end with <s>, </s>");
        }
    }

    if (diag.size() > 0)
        corpus.push_back(diag);
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << sd.size() << " types\n";
    return corpus;
}

void get_string_and_its_id(const string &filename, const pair<int, int>& columids, const string& save_to_filename)
{
    ifstream in(filename);
    string line;

    if (save_to_filename.size() == 0)
        throw("need to have filename to write a mapping of string id to the string");

    stId2String<string> id2string;

    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            continue;

        Sentence source;
        string diagid = ReadStringWithItsId(line, source, id2string, columids);
        if (diagid.size() == 0)
            continue;
    }

    ofstream on(save_to_filename);
    boost::archive::text_oarchive oa(on);
    oa << id2string;
}

/**
columnids : <int,int>
For example, the input has
<diag_id> ||| <turn_id> ||| <user_intput> ||| <response> ||| <response_id> ||| <typical response>
setting columnids as <2,3> will output
<user_input> and <response>
*/
Corpus read_corpus(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS, bool backofftounk, const pair<int, int>& columnids)
{
    ifstream in(filename);
    string line;

    Corpus corpus;
    Dialogue diag;
    string prv_diagid = "-1";
    int lc = 0, stoks = 0, ttoks = 0;

    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            continue;
        ++lc;
        Sentence source, target;
        string diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, backofftounk, kSRC_SOS, kSRC_EOS, columnids, make_pair<bool, bool>(true, true));
        if (diagid.size() == 0)
            continue;

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

Corpus read_corpus(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength, bool backofftounk, bool bcharacter)
{
    ifstream in(filename);
    string line;

    Corpus corpus;
    Dialogue diag;
    string prv_diagid = "-1";
    int lc = 0, stoks = 0, ttoks = 0;

    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            continue;
        ++lc;
        Sentence source, target;
        string diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, backofftounk, kSRC_SOS, kSRC_EOS, bcharacter);
        if (diagid.size() == 0)
            continue;

        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }
        if (source.size() > maxSentLength)
        {
            source.resize(maxSentLength - 1);
            source.push_back(kSRC_EOS);
        }
        if (target.size() > maxSentLength)
        {
            target.resize(maxSentLength - 1);
            target.push_back(kSRC_EOS);
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

Corpus read_corpus(ifstream & in, Dict& sd, int kSRC_SOS, int kSRC_EOS, long part_size)
{
    string line;

    Corpus corpus;
    Dialogue diag;
    string prv_diagid = "-1";
    int lc = 0, stoks = 0, ttoks = 0;

    long iln = 0;
    while (getline(in, line) && iln < part_size) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, target;
        string diagid;

        diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, false, kSRC_SOS, kSRC_EOS, false);
        if (diagid == "")
            continue;

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
            cerr << "Sentence in " << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
        
        iln++;
    }

    if (diag.size() > 0)
        corpus.push_back(diag);
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << sd.size() << " types\n";
    return corpus;
}

/**
phyid2logicid : the physical id in the training data file is mapped to a logic id. 
*/
Corpus read_corpus(ifstream & in, Dict& sd, int kSRC_SOS, int kSRC_EOS, long part_size, const pair<int, int>& columids, const pair<bool, bool>& use_dict, unordered_map<int,int>& phyid2logicid)
{
    string line;

    Corpus corpus;
    Dialogue diag;
    string prv_diagid = "-1";
    int lc = 0, stoks = 0, ttoks = 0;

    long iln = 0;
    while (getline(in, line) && iln < part_size) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, target;
        string diagid;

        diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, true, kSRC_SOS, kSRC_EOS, columids, use_dict);
        if (diagid == "")
            continue;

        if (phyid2logicid.size() > 0)
        {
            Sentence newtarget; 
            for (auto& p : target){
                if (phyid2logicid.find(p) != phyid2logicid.end())
                    newtarget.push_back(phyid2logicid[p]);
                else
                    throw("cannot find a physical id that has a logic id");
            }
            target = newtarget;
        }

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
            cerr << "Sentence in " << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }

        iln++;
    }

    if (diag.size() > 0)
        corpus.push_back(diag);
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << sd.size() << " types\n";
    return corpus;
}

/**
the data contains triplet
user input ||| response ||| addition input such as intention
*/
TupleCorpus read_tuple_corpus(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS, Dict& td, int kTGT_SOS, int kTGT_EOS, int maxSentLength)
{
    ifstream in(filename);
    string line;

    TupleCorpus corpus;
    TupleDialogue diag;
    vector<Dict*> vDict;
    vDict.push_back(&sd);
    vDict.push_back(&td);
    vDict.push_back(&sd);

    int prv_diagid = -1;
    int lc = 0, stoks = 0, ttoks = 0;
    long min_diag_id = 99999;
    while (getline(in, line)) 
    {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence i_source, i_answer, i_query;
        vector<Sentence*> source;
        source.push_back(&i_source);
        source.push_back(&i_answer);
        source.push_back(&i_query);
        int diagid = MultiTurnsReadSentence(line, source, vDict);
        if (diagid == -1)
            break;

        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }

        SentenceTuple stuple = make_triplet_sentence(i_source, i_answer, i_query);
        diag.push_back(stuple);

        stoks += source[0]->size();
        ttoks += source[1]->size();
    }

    if (diag.size() > 0)
        corpus.push_back(diag);
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
    return corpus;
}

SentenceTuple make_triplet_sentence(const Sentence& m1, const Sentence& m2, const Sentence& m3)
{
    return make_triplet<Sentence>(m1, m2, m3);
}

string MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, bool backofftounk, int kSRC_SOS, int kSRC_EOS, bool bcharacter)
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::string diagid, turnid;

    std::vector<int>* v = s;

    if (line.length() == 0)
        return "";

    in >> diagid;
    trim(diagid);
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting diagid" << endl;
        return "";
    }

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting turn id" << endl;
        return "";
    }

    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            d = td; v = t;
            continue;
        }
        /// if character need to add blank before and after string, also seperate chacter with blank
        if (bcharacter && word != "<s>" & word != "</s>")
        {
            v->push_back(d->Convert(" ", backofftounk));
            for (size_t k = 0; k < word.size();k++)
                v->push_back(d->Convert(boost::lexical_cast<string>(word[k]), backofftounk));
        }
        else
        {
            if (word == "</s>" && bcharacter)
                v->push_back(d->Convert(" ", backofftounk));
            v->push_back(d->Convert(word, backofftounk));
        }
    }

    return diagid;
}

string MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, bool backofftounk, int kSRC_SOS, int kSRC_EOS, const pair<int, int>& columnids, const pair<bool, bool>& use_dict)
{
    int cid = 0;
    int work_on_id = columnids.first;
    bool to_use_dict = use_dict.first;
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::string diagid, turnid;

    std::vector<int>* v = s;

    if (line.length() == 0)
        return "";

    in >> diagid;
    trim(diagid);
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting diagid" << endl;
        return "";
    }
    cid++;

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting turn id" << endl;
        return "";
    }
    cid++;

    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            d = td; v = t;
            cid++;
            work_on_id = columnids.second;
            to_use_dict = use_dict.second;
            continue;
        }
        if (cid == work_on_id)
        {
            if (to_use_dict)
                v->push_back(d->Convert(word, backofftounk));
            else
                v->push_back(boost::lexical_cast<int>(word));
        }
    }

    return diagid;
}

/// read string and its id. their positions are decided in columnids
/// columnids is a pair saving <string position, id position>
string ReadStringWithItsId(const std::string& line, std::vector<int>& s, stId2String<string>& sd, const pair<int, int>& columnids)
{
    int cid = 0;
    int string_position = columnids.first;
    int id_position = columnids.second;
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    std::string diagid, turnid;

    if (line.length() == 0)
        return "";

    in >> diagid;
    trim(diagid);
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting diagid" << endl;
        return "";
    }
    cid++;

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting turn id" << endl;
        return "";
    }
    cid++;

    string str = "";
    vector<string> wid;
    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            cid++;
            continue;
        }
        if (cid == string_position)
        {
            str = str + " " + word;
        }
        if (cid == id_position)
            wid.push_back(word); 
    }

    if (wid.size() != 1)
    {
        for (auto& p : wid)
            cout << " " << p;
        throw("ReadStringWithItsId : word position column has multiple ids");
    }
    int id = boost::lexical_cast<int>(wid[0]);
    s.push_back(id);
    sd.Convert(id, str);
    return diagid;
}

string MultiTurnsReadSentencePairWithClassId(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int>* cls, int kSRC_SOS, int kSRC_EOS)
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::string diagid, turnid;

    std::vector<int>* v = s;

    if (line.length() == 0)
        return "";

    in >> diagid;
    trim(diagid);
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| <src> || <tgt> ||| <classid>" << endl;
        cerr << "expecting diagid" << endl;
        return "";
    }

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| <src> || <tgt> ||| <classid>" << endl;
        cerr << "expecting turn id" << endl;
        return "";
    }

    int septimes = 0;
    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            if (septimes == 0)
            {
                d = td; v = t;
            }
            septimes++;
            continue;
        }

        if (septimes <= 1)
            v->push_back(d->Convert(word, true));
        else
            cls->push_back(boost::lexical_cast<int>(word));
    }

    return diagid;
}

int MultiTurnsReadSentence(const std::string& line,
    vector<std::vector<int>*> s,
    vector<Dict*> sd)
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd[0];
    std::string diagid, turnid;

    std::vector<int>* v = s[0];

    if (line.length() == 0)
        return -1;

    in >> diagid;
    trim(diagid);
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt ||| additional " << endl;
        cerr << "expecting diagid" << endl;
        return -1;
    }

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt ||| additional " << endl;
        cerr << "expecting turn id" << endl;
        return -1;
    }

    size_t kk = 0;
    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            ++kk;
            d = sd[kk]; v = s[kk];
            continue;
        }

        v->push_back(d->Convert(word));
    }

    int res;

    res = boost::lexical_cast<int>(diagid);

    return res;
}

int MultiTurnsReadSentencePair(const std::wstring& line, std::vector<int>* s, WDict* sd, std::vector<int>* t, WDict* td, bool appendSBandSE, int kSRC_SOS, int kSRC_EOS)
{
    std::wistringstream in(line);
    std::wstring word;
    std::wstring sep = L"|||";
    WDict* d = sd;
    std::wstring diagid, turnid;

    std::vector<int>* v = s;

    if (line.length() == 0)
        return -1;

    in >> diagid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting diagid" << endl;
        return -1;
    }

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting turn id" << endl;
        return -1;
    }

    if (appendSBandSE)
        v->push_back(kSRC_SOS);
    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            if (appendSBandSE)
                v->push_back(kSRC_EOS);
            d = td; v = t;
            if (appendSBandSE)
                v->push_back(kSRC_SOS);
            continue;
        }
        v->push_back(d->Convert(word));
    }
    if (appendSBandSE)
        v->push_back(kSRC_EOS);
    int res;

    wstringstream(diagid) >> res;
    return res;
}

NumTurn2DialogId get_numturn2dialid(Corpus corp)
{
    NumTurn2DialogId info;

    int id = 0;
    for (auto p : corp)
    {
        size_t d_turns = p.size();
        info.mapNumTurn2DialogId[d_turns].push_back(id++);
    }
    for (auto p : info.mapNumTurn2DialogId)
    {
        info.vNumTurns.push_back(p.first);
    }
    return info;
}

NumTurn2DialogId get_numturn2dialid(TupleCorpus corp)
{
    NumTurn2DialogId info;

    int id = 0;
    for (auto p : corp)
    {
        size_t d_turns = p.size();
        info.mapNumTurn2DialogId[d_turns].push_back(id++);
    }
    for (auto p : info.mapNumTurn2DialogId)
    {
        info.vNumTurns.push_back(p.first);
    }
    return info;
}


/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
/// this assumes same length
Expression shuffle_data(Expression src, unsigned nutt, unsigned feat_dim, unsigned slen)
{
    Expression i_src = reshape(src, {nutt * slen * feat_dim});

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

/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
/// this assumes different source length
/// the result vector for each element doesn't have redundence, i.e., all elements are valid.
vector<Expression> shuffle_data(Expression src, unsigned nutt, unsigned feat_dim, const vector<unsigned>& v_slen)
{
    /// the input data is arranged into a big matrix, assuming same length of utterance
    /// but they are different length
    unsigned slen = *std::max_element(v_slen.begin(), v_slen.end());

    Expression i_src = reshape(src, { nutt * slen * feat_dim });

    int stride = nutt * feat_dim;
    vector<Expression> i_all_spk;
    for (size_t k = 0; k < nutt; k++)
    {
        vector<Expression> i_each_spk;
        for (size_t t = 0; t < v_slen[k]; t++)
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
    return i_all_spk;
}

void convertHumanQuery(const std::string& line, std::vector<int>& t, Dict& td)
{
    std::istringstream in(line);
    std::string word;
    t.clear();

    while (in) {
        in >> word;
        if (!in) break;
        t.push_back(td.Convert(word, true));
    }
}

void convertHumanQuery(const std::wstring& line, std::vector<int>& t, WDict& td)
{
    std::wistringstream in(line);
    std::wstring word;

    t.clear();

    while (in) {
        in >> word;
        if (!in) break;
        t.push_back(td.Convert(word, true));
    }
}

FBCorpus read_facebook_qa_corpus(const string &filename, size_t& diag_id, Dict& sd)
{
    ifstream in(filename);
    generator gen;

    string line;
    int turnid;

    diag_id = -1;
    FBCorpus corpus;
    FBDialogue diag;
    FBTurns turns;
    StatementsQuery sq;
    vector<Sentence> statements;

    int prv_turn = 9999, lc = 0, stoks = 0, ttoks = 0;

    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, query, target;
        vector<string> vstr; 
        string newline;
        string ans; 

        if (find(line.begin(), line.end(), '?') != line.end())
        {
            /// check if question
            boost::split(vstr, line, boost::is_any_of("\t"));
            ans = vstr[1];
            
            newline = vstr[0];
            std::replace(newline.begin(), newline.end(), '?', ' ');

            turnid = read_one_line_facebook_qa(newline, query, sd);
            sq = make_pair(statements, query);

            Sentence sans;
            sans.push_back(sd.Convert(ans));
            turns = make_pair(sq, sans);

            ttoks++;
            diag.push_back(turns);
            statements.clear();
        }
        else
        {
            newline = line;
            std::replace(newline.begin(), newline.end(), '.', ' '); 
            turnid = read_one_line_facebook_qa(newline, source, sd);
            statements.push_back(source);
        }

        if (turnid < prv_turn)
        {
            diag_id++;

            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
        }
        prv_turn = turnid;
    }

    cerr << lc << " lines & " << diag_id << " dialogues & " << ttoks << " questions " << endl; 
    return corpus;
}

int read_one_line_facebook_qa(const std::string& line, std::vector<int>& v, Dict& sd)
{
    std::istringstream in(line);
    std::string word;
    std::string turnid;

    if (line.length() == 0)
        return -1;

    in >> turnid;

    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        v.push_back(sd.Convert(word));
    }

    return boost::lexical_cast<int, string>(turnid);
}

Expression vec2exp(const vector<cnn::real>& v_data, ComputationGraph& cg)
{
    Expression iv;
    iv = input(cg, { (unsigned) v_data.size() }, &v_data);

    return iv;
}

vector<cnn::real> read_embedding(const string& line, Dict& sd, int & index)
{
    std::istringstream in(line);
    std::string word;

    size_t i = 0;
    int id = -1;
    vector<cnn::real> v_data;

    while (in) {
        in >> word;
        trim(word);
        if (!in) break;

        if (i == 0)
        {
            if (sd.Contains(word))
                id = sd.Convert(word);
            else
                break;
        }
        else
        {
            v_data.push_back(boost::lexical_cast<cnn::real>(word));
        }
        i++;
    }

    index = id;
    return v_data;
}

void read_embedding(const string& embedding_fn, Dict& sd, map<int, vector<cnn::real>> & vWordEmbedding)
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

    // generate word embedding for unknown words by averaging 100 words
    vector<cnn::real> iv = vWordEmbedding.begin()->second;
    size_t tk = 1;
    for (auto& p : vWordEmbedding)
    {
        std::transform(iv.begin(), iv.end(), p.second.begin(), iv.begin(), std::plus<cnn::real>());
        tk++;
        if (tk > 100)
            break;
    }
    std::transform(iv.begin(), iv.end(), iv.begin(), std::bind1st(std::multiplies<cnn::real>(), 1.0/tk));

    // back off to a word embedding for unk for those words that don't have embedding
    for (auto &p : sd.GetWordList())
    {
        if (vWordEmbedding.find(sd.Convert(p)) != vWordEmbedding.end())
            continue;
        vWordEmbedding[sd.Convert(p)] = iv;
    }
}

string builder_flavour(variables_map vm)
{
    string flavour = "rnn";
    if (vm.count("lstm"))	flavour = "lstm";
    else if (vm.count("rnn_elu"))	flavour = "rnn_elu";
    else if (vm.count("gru"))	flavour = "gru";
    else if (vm.count("dglstm"))	flavour = "dglstm";
    else if (vm.count("dglstm-dnn")) flavour = "dnn";
    else if (vm.count("builder")>0)
    {
        flavour = vm["builder"].as<string>();
    }

    return flavour;
}

vector<int> remove_first_and_last(const vector<int>& rep)
{
    Sentence trimedrep(rep.size() - 2);
    /// remove <s> and </s>
    std::copy(rep.begin() + 1, rep.end() - 1, trimedrep.begin());
    return trimedrep;
}

void DataReader::read_corpus(Dict& sd, int kSRC_SOS, int kSRC_EOS, long part_size)
{
    string line;

    m_Corpus.clear();

    Dialogue diag;
    string prv_diagid = "-1";
    int lc = 0, stoks = 0, ttoks = 0;

    long iln = 0;
    while (getline(m_ifs, line) && iln < part_size) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, target;
        string diagid;

        diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, false, kSRC_SOS, kSRC_EOS, false);
        if (diagid == "")
            continue;

        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                m_Corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }
        diag.push_back(SentencePair(source, target));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS)) {
            cerr << "Sentence in " << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }

        iln++;
    }

    if (diag.size() > 0)
        m_Corpus.push_back(diag);
    cerr << "from corpus " << m_Filename << ": " << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << sd.size() << " types\n";
}

const unsigned int PRIMES[] = { 108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807 };

const unsigned int PRIMES_SIZE = sizeof(PRIMES) / sizeof(PRIMES[0]);

vector<unsigned int> hashing(const vector<int>& obs, int direct_order, int hash_size)
{
    int a = 0;
    vector<unsigned int> hash(direct_order * obs.size(), 0);

    for (int k = 0; k < obs.size(); k++)
    {
        int offset = k * direct_order;
        for (a = 0; a < direct_order; a++) {
            int b = 0;
            /// this performs convolution operation
            hash[a + offset] = PRIMES[0] * PRIMES[1];
            for (b = 0; b <= a; b++) hash[a + offset] += PRIMES[(a*PRIMES[b] + b) % PRIMES_SIZE] * (unsigned long long)((b+k>obs.size() - 1)?0:obs[b + k] + 1);	//update hash value based on words from the history
            hash[a + offset] = hash[a + offset] % hash_size;
        }
    }

    return hash;
}
