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
