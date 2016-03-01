#pragma once

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <math.h>
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
#include "cnn/math.h"
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/serialization/map.hpp>

using namespace cnn;
using namespace std;
using namespace boost::program_options;

typedef vector<cnn::real> LossStats;

/** classical language model
*/
typedef std::map<int, cnn::real>  tUnigram;  /// unigram
typedef std::map<pair<int, int>, cnn::real>  tBigram;  /// bigram
typedef std::map<int, int> tUniCount;
typedef std::map<pair<int,int>, int> tBiCount;
class nGram
{
protected:
    tUnigram lgUniLM;  /// unigram
    tBigram  lgBiLM;  /// bigram
    tUniCount unicnt;
    tBiCount  bicnt;

    unsigned long nwords;
    unsigned long vocab_size;

    cnn::real interpolation_wgt; 
    int NgramOrder;
    string model_filename;

public:

    nGram()
    {
        interpolation_wgt = 0.9;
        vocab_size = 0;
        nwords = 0;
    }

    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive& ar, const unsigned int version) {
        ar & nwords;
        ar & vocab_size;
        ar & BOOST_SERIALIZATION_NVP(lgUniLM);
        ar & BOOST_SERIALIZATION_NVP(lgBiLM);
    }

    /// sentence log-likelihood
    cnn::real GetSentenceLL(const vector<string> & refTokens, Dict& sd)
    {
        int prv_wrd = -1;
        cnn::real prob = 0.0;
        for (int n = 0; n < refTokens.size(); n++)
        {
            int wrd = sd.Convert(refTokens[n]);
            if (n == 0)
            {
                prv_wrd = wrd;
                prob = lgUniLM[wrd];
                continue;
            }

            pair<int, int> bins = make_pair(prv_wrd, wrd);
            if (lgBiLM.find(bins) == lgBiLM.end())
                prob += log(interpolation_wgt) + lgUniLM[wrd];
            else
                prob += log(exp(lgBiLM[bins]) * (1.0 - interpolation_wgt) + interpolation_wgt * exp(lgUniLM[wrd]));
        }
        prob /= refTokens.size();
        return prob;
    }

    cnn::real GetSentenceLL(const Sentence & refTokens, cnn::real interpolation_wgt)
    {
        int prv_wrd = -1;
        cnn::real prob = 0.0;
        for (int n = 0; n < refTokens.size(); n++)
        {
            int wrd = refTokens[n]; 
            if (n == 0)
            {
                prv_wrd = wrd;
                prob = lgUniLM[wrd];
                continue;
            }

            pair<int, int> bins = make_pair(prv_wrd, wrd);
            if (lgBiLM.find(bins) == lgBiLM.end())
                prob += lgUniLM[wrd];
            else
                prob += log(exp(lgBiLM[bins]) * (1.0 - interpolation_wgt) + interpolation_wgt * exp(lgUniLM[wrd]));
            prv_wrd = wrd;
        }
        if (refTokens.size() == 0)
            prob = LZERO;
        else
            prob /= refTokens.size();
        return prob;
    }

    void Initialize(const variables_map & vm)
    {
        NgramOrder = 1; // fix to bigram
        model_filename = vm["parameters"].as<string>(); 
    }

    void SaveModel(const string& ext)
    {
        string fname = model_filename + ext;
        ofstream on(fname);
        boost::archive::text_oarchive oa(on);
        oa << *this;
    }

    void LoadModel(const string& ext)
    {
        string fname = model_filename + ext;
        ifstream in(fname);
        if (!in.is_open())
        {
            return;
        }
        boost::archive::text_iarchive ia(in);
        ia >> *this;
    }

    void SaveModel()
    {
        string fname = model_filename;
        ofstream on(fname);
        boost::archive::text_oarchive oa(on);
        oa << *this;
    }

    void LoadModel()
    {
        string fname = model_filename;
        ifstream in(fname);
        if (!in.is_open())
        {
            cerr << "cannot open " << fname << endl;
            throw("cannot open " + fname);
        }
        boost::archive::text_iarchive ia(in);
        ia >> *this;
    }

public:

    void Clear()
    {
        nwords = 0;
        lgUniLM.clear();
        lgBiLM.clear();
        unicnt.clear();
        bicnt.clear();
    }

    void UpdateNgramCounts(const vector<string> & tokens, int order, Dict& sd)
    {
        vocab_size = sd.size();
        if (tokens.size() < order)
            return;

        int n = order;
        if (n > 2)
            throw("only support bigram");
        for (int i = 0; i < tokens.size() - n; i++)
        {
            vector<int> sb;
            for (int j = 0; j <= n; j++)
            {
                sb.push_back(sd.Convert(tokens[i + j]));
            }

            if (n == 1)
            {
                int pb;
                pb = sb[0];
                if (unicnt.find(pb) == unicnt.end())
                {
                    unicnt[pb] = 1;
                }
                else
                {
                    unicnt[pb]++;
                }
                nwords++;
            }
            if (n == 2)
            {
                pair<int, int> pb; 
                pb = make_pair(sb[0], sb[1]);
                if (bicnt.find(pb) == bicnt.end())
                {
                    bicnt[pb] = 1;
                }
                else
                {
                    bicnt[pb]++;
                }
            }
        }
    }

    void UpdateNgramCounts(const Sentence & tokens, int order, Dict& sd)
    {
        vocab_size = sd.size();
        if (tokens.size() < order)
            return;

        int n = order;
        if (n > NgramOrder)
            throw("only support bigram");
        for (int i = 0; i < tokens.size() - n; i++)
        {
            vector<int> sb;
            for (int j = 0; j <= n; j++)
            {
                sb.push_back(tokens[i + j]);
            }

            if (n == 0)
            {
                int pb;
                pb = sb[0];
                if (unicnt.find(pb) == unicnt.end())
                {
                    unicnt[pb] = 1;
                }
                else
                {
                    unicnt[pb]++;
                }
                nwords++;
            }
            if (n == 1)
            {
                pair<int, int> pb;
                pb = make_pair(sb[0], sb[1]);
                if (bicnt.find(pb) == bicnt.end())
                {
                    bicnt[pb] = 1;
                }
                else
                {
                    bicnt[pb]++;
                }
            }
        }
    }

    void ComputeNgramModel()
    {
        /// for smoothing
        for (long i = 0; i < vocab_size; i++)
            lgUniLM[i] = -log(vocab_size + nwords);

        /// add-one smoothing of unigram
        for (auto & p : unicnt)
        {
            cnn::real pb = log(p.second + 1) - log(nwords + vocab_size);
            lgUniLM[p.first] = pb;
        }

        /// assert
        cnn::real prb = 0.0;
        for (long i = 0; i < vocab_size; i++)
            prb += exp(lgUniLM[i]);

        for (auto & p : bicnt)
        {
            int src, tgt;
            src = p.first.first;
            tgt = p.first.second;

            int cnt = p.second;
            int srccnt = unicnt[src];

            lgBiLM[make_pair(src, tgt)] = log(cnt) - log(srccnt);
        }
    }

};
