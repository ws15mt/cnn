#pragma once

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

using namespace cnn;
using namespace std;

typedef vector<cnn::real> FVector;
typedef vector<FVector>   FMatrix;
typedef vector<FMatrix>   FCorpus;
typedef vector<FCorpus*>  FCorpusPointers;

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Dialogue;
typedef vector<Dialogue> Corpus;

/**
usually packs a matrix with real value element
this truncates both source and target 
@mbsize : number of samples
@nutt : number of sentences to process in parallel
*/
vector<vector<Expression>> pack_obs(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& rand_stt);

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs_uttfirst(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& rand_stt);

void get_same_length_dialogues(Corpus corp, size_t nbr_dialogues, size_t &stt_dialgoue_id, vector<bool>& used, vector<Dialogue>& selected);

int MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td);

Corpus read_corpus(const string &filename, unsigned& min_diag_id, Dict& sd, int kSRC_SOS, int kSRC_EOS);

/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
Expression shuffle_data(Expression src, size_t nutt, size_t feat_dim, size_t slen);

