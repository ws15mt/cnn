#pragma once

#include <map>
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/dict.h"
#include <boost/program_options/variables_map.hpp>
#include <boost/thread.hpp>
#include <fstream>

using namespace cnn;
using namespace std;
using namespace boost::program_options;

typedef vector<cnn::real> FVector;
typedef vector<FVector>   FMatrix;
typedef vector<FMatrix>   FCorpus;
typedef vector<FCorpus*>  FCorpusPointers;

typedef vector<int> Sentence;
typedef vector<Sentence> Sentences;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Dialogue;
typedef vector<Dialogue> Corpus;

typedef pair<Sentence, int> SentenceWithId;
typedef pair<Sentence, SentenceWithId> SentencePairAndClassId;
typedef vector<SentencePairAndClassId> DialogueWithClassId;
typedef vector<DialogueWithClassId> CorpusWithClassId;

/// for parallel processing of data
typedef vector<SentencePair> PTurn;  /// a turn consits of sentences pairs from difference utterances [#sentences]
typedef vector<PTurn> PDialogue;  /// a parallel dialogue consists of turns from each parallel dialogue [#turns][#sentences]

template<class T>
struct triplet
{
    T first;
    T middle;
    T last;
};

template<class T>
triplet<T> make_triplet(const T &m1, const T &m2, const T &m3)
{
    triplet<T> ans;
    ans.first = m1;
    ans.middle = m2;
    ans.last = m3;
    return ans;
};

typedef triplet<Sentence> SentenceTuple;

typedef vector<SentenceTuple> TupleDialogue;
typedef vector<TupleDialogue> TupleCorpus;

SentenceTuple make_triplet_sentence(const Sentence& m1, const Sentence& m2, const Sentence& m3);

/// for parallel processing of data
typedef vector<SentencePair> PTurn;  /// a turn consits of sentences pairs from difference utterances
typedef vector<PTurn> PDialogue;  /// a dialogue consists of many turns
typedef vector<PDialogue> PCorpus; /// a parallel corpus consists of many parallel dialogues

/// save the number of turns to dialogue id list
typedef struct{
    vector<int> vNumTurns;  /// vector saving number of turns to be accessed, can shuffle this vector so that the access of dialogues are randomized
    map<int, vector<int>> mapNumTurn2DialogId;
} NumTurn2DialogId;

typedef pair<vector<Sentence>, Sentence> StatementsQuery;
typedef pair<StatementsQuery, Sentence> FBTurns;
typedef vector<FBTurns> FBDialogue;
typedef vector<FBDialogue> FBCorpus;


/**
usually packs a matrix with real value element
this truncates both source and target 
@mbsize : number of samples
@nutt : number of sentences to process in parallel
*/
vector<vector<Expression>> pack_obs(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& rand_stt);

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs_uttfirst(FCorpusPointers raw, unsigned mbsize, ComputationGraph& cg, const vector<size_t>& randstt);

/// read one line of word embeddings
/// dataformat is as follows
/// the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 -0.6566 0.27843 -0.14767 -0.55677 
/// return an expression and also an index value of this word according to a dictionary sd
vector<cnn::real> read_embedding(const string& line, Dict& sd, int & index);

/// read word embedding from a file with format above
/// for words that are not in the word embeddin file, use the first 100 word embeddings to construct a representation for those words
void read_embedding(const string& embedding_fn, Dict& sd, map<int, vector<cnn::real>> & vWordEmbedding);

Expression vec2exp(const vector<cnn::real>& v_data, ComputationGraph& cg);

//////////////////////////////////////////////////////////////////////////////////////////////
/// pading and reading data
//////////////////////////////////////////////////////////////////////////////////////////////

/// padding with eos symbol
PDialogue padding_with_eos(const PDialogue& v_diag, int padding_symbol, const std::vector<bool>& padding_to_back);
Sentences padding_with_eos(const Sentences& v_sent, int padding_symbol, bool  padding_to_the_back);

/// return the index of the selected dialogues
vector<int> get_same_length_dialogues(Corpus corp, int nbr_dialogues, size_t &min_nbr_turns, vector<bool>& used, PDialogue& selected, NumTurn2DialogId& info);

/**
read corpus
@bcharacter : read data in character level. default is false, which is word-level.
*/
Corpus read_corpus(const string &filename, unsigned& min_diag_id, WDict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength = 10000, bool appendBSandES = false);
int MultiTurnsReadSentencePair(const std::wstring& line, std::vector<int>* s, WDict* sd, std::vector<int>* t, WDict* td, bool appendSBandSE = false, int kSRC_SOS = -1, int kSRC_EOS = -1);
Corpus read_corpus(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength = 10000, bool appendBSandES = false, bool bcharacter = false);
Corpus read_corpus(ifstream&, Dict& sd, int kSRC_SOS, int kSRC_EOS, long part_size);
CorpusWithClassId read_corpus_with_classid(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS);
Corpus read_corpus(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS, bool backofftounk, const pair<int, int>& columnids);
Corpus read_corpus(ifstream & in, Dict& sd, int kSRC_SOS, int kSRC_EOS, long part_size, const pair<int, int>& columids, const pair<bool, bool>& ues_dict, unordered_map<int, int>& ph2logic);

/** get id and its string */
void get_string_and_its_id(const string &filename, const pair<int, int>& columids, const string& save_to_filename);

/**
read sentence pair in one line, with seperaotr |||
@bcharacter : read data in character level, default is false, which is word-level
*/
string MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, bool appendSBandSE = false, int kSRC_SOS = -1, int kSRC_EOS = -1, bool bcharacter = false);

/**
read sentences pair with class id at the end
*/
string MultiTurnsReadSentencePairWithClassId(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int>* cls, int kSRC_SOS, int kSRC_EOS);

/**
read sentences from corpus. assume that the separator of content is "|||". 
which columns to be selected are decided in columnids;
whether to apply a dictionary lookup is decided in use_dict. 
*/
string MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, bool backofftounk, int kSRC_SOS, int kSRC_EOS, const pair<int, int>& columnids, const pair<bool, bool>& use_dict);

/// read string and its id. their positions are decided in columnids
/// columnids is a pair saving <string position, id position>
string ReadStringWithItsId(const std::string& line, std::vector<int>& s, stId2String<string>& sd, const pair<int, int>& columnids);

/**
read corpus with triplet
user input ||| answer ||| intention/question
*/
TupleCorpus read_tuple_corpus(const string &filename, Dict& sd, int kSRC_SOS, int kSRC_EOS, Dict& td, int kTGT_SOS, int kTGT_EOS, int maxSentLength);
int MultiTurnsReadSentence(const std::string& line,
    vector<std::vector<int>*> s,
    vector<Dict*> sd);

NumTurn2DialogId get_numturn2dialid(Corpus corp);
NumTurn2DialogId get_numturn2dialid(TupleCorpus corp);

void flatten_corpus(const Corpus& corpus, vector<Sentence>& sentences, vector<Sentence>& responses);
void flatten_corpus(const CorpusWithClassId& corpus, vector<Sentence>& sentences, vector<SentenceWithId>& response);

/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
Expression shuffle_data(Expression src, unsigned nutt, unsigned feat_dim, unsigned slen);

vector<Expression> shuffle_data(Expression src, unsigned nutt, unsigned feat_dim, const vector<unsigned>& v_slen);

void convertHumanQuery(const std::string& line, std::vector<int>& t, Dict& td);

void convertHumanQuery(const std::wstring& line, std::vector<int>& t, WDict& td);

std::wstring utf8_to_wstring(const std::string& str);

std::string wstring_to_utf8(const std::wstring& str);


/// utiles to read facebook data
int read_one_line_facebook_qa(const std::string& line, std::vector<int>& v, Dict& sd);
FBCorpus read_facebook_qa_corpus(const string &filename, size_t & diag_id, Dict& sd);

/**
return flavour of a builder in string
*/
std::string builder_flavour(variables_map vm);

/// remove the firest and the last element
vector<int> remove_first_and_last(const vector<int>& rep);

/**
using own thread to read data into a host memory
*/
class DataReader{
private : 
    boost::thread m_Thread; 
    std::ifstream m_ifs;
    string        m_Filename; /// the file name
    Corpus        m_Corpus;   /// the corpsu;

    void read_corpus(Dict& sd, int kSRC_SOS, int kSRC_EOS, long part_size);

public:
    DataReader(const string& train_filename)
    {
        m_Filename = train_filename; 
        m_ifs.open(train_filename);
    }

    ~DataReader() { m_ifs.close();  }

    void restart()
    {
        m_ifs.close();
        m_ifs.open(m_Filename);
    }

    void start(Dict& sd, int kSRC_SOS, int kSRC_EOS, long part_size)
    {
        m_Thread = boost::thread(&DataReader::read_corpus, this, sd, kSRC_SOS, kSRC_EOS, part_size);
    }

    Corpus corpus()
    {
        return m_Corpus;
    }

    void join() { m_Thread.join(); }
    void detach() { m_Thread.detach();  }
};

/// for each sequence, return a hash with order direct_order
/// this gives a vector with size of direct_order * obs.size()
/// which consists of unigram, if direct_order == 1, 
/// bigram if direct_order == 2, etc. 
/// e.g., direct_order = 2, the output includes both unigram and bigram hashing
vector<unsigned int> hashing(const vector<int> & obs, int direct_order, int hash_size);


