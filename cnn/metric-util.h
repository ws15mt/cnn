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
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

using namespace cnn;
using namespace std;
using namespace boost::program_options;

typedef vector<cnn::real> LossStats;

class BleuMetric
{
private:
    LossStats m_allStats;

    int NgramOrder;
    int m_refIndex;
    int m_hypIndex;
    int m_matchIndex;

public:

    BleuMetric()
    {
    }

    ~BleuMetric()
    {}
    
    void AccumulateScore(const vector<string> & refTokens, const vector<string> & hypTokens)
    {
        LossStats stats = GetStats(refTokens, hypTokens);

        size_t k = 0;
        for (auto &p : stats){
            m_allStats[k++] += p;
        }
    }

    string GetScore()
    {
        cnn::real precision = Precision(m_allStats);
        cnn::real bp = BrevityPenalty(m_allStats);

        cnn::real score = 100.0*precision*bp;
        return boost::lexical_cast<string>(score);
    }

    cnn::real GetSentenceScore(const vector<string> & refTokens, const vector<string> & hypTokens)
    {
        LossStats stats = GetStats(refTokens, hypTokens);
        cnn::real precision = Precision(stats);
        cnn::real bp = BrevityPenalty(stats);

        cnn::real score = 100.0*precision*bp;
        return score;
    }

    LossStats GetStats(const vector<string> & refTokens, const vector<string> & hypTokens)
    {
        vector<string> lcRefTokens, lcHypTokens;
        lcRefTokens = refTokens;
        lcHypTokens = hypTokens;

        LossStats stats;
        stats.resize(1 + 2 * NgramOrder, 0.0);
        stats[m_refIndex] = ((cnn::real)lcRefTokens.size());
        for (int j = 0; j < NgramOrder; j++)
        {
            map<string, int> refCounts = GetNgramCounts(lcRefTokens, j);
            map<string, int> hypCounts = GetNgramCounts(lcHypTokens, j);

            int overlap = 0;
            for (map<string, int>::iterator e = hypCounts.begin(); e != hypCounts.end(); e++)
            {
                string ngram = e->first;
                int hypCount = e->second;
                int refCount = refCounts.count(ngram);
                overlap += min<int>(hypCount, refCount);
            }
            stats[m_hypIndex + j] = ((cnn::real)max<int>(0, lcHypTokens.size() - j));
            stats[m_matchIndex + j] = ((cnn::real)overlap);
        }
        return stats;
    }

    void Initialize(const variables_map & vm)
    {
        NgramOrder = vm["ngram_order"].as<int>(); ///default  4;
        m_allStats.resize(1 + 2 * NgramOrder, 0.0);

        m_refIndex = 0;
        m_hypIndex = m_refIndex + 1;
        m_matchIndex = m_hypIndex + NgramOrder;
    }

    void Initialize(int ngramorder = 4)
    {
        NgramOrder = ngramorder; 
        m_allStats.resize(1 + 2 * NgramOrder, 0.0);

        m_refIndex = 0;
        m_hypIndex = m_refIndex + 1;
        m_matchIndex = m_hypIndex + NgramOrder;
    }

    cnn::real BrevityPenalty(LossStats stats)
    {
        cnn::real refLen = stats[m_refIndex];
        cnn::real hypLen = stats[m_hypIndex];
        if (hypLen >= refLen)
        {
            return 1.0;
        }
        cnn::real bp = exp(1.0 - refLen / hypLen);
        return bp;
    }

private:
    std::map<string, int> GetNgramCounts(const vector<string> & tokens, int order)
    {
        map<string, int> counts;

        int n = order;

        if (tokens.size() < n + 1)
            return counts; 
        
        for (int i = 0; i < tokens.size() - n; i++)
        {
            string sb;
            for (int j = 0; j <= n; j++)
            {
                if (j > 0)
                {
                    sb += " ";
                }
                sb += tokens[i + j];
            }
            string ngram = sb;
            if (counts.find(ngram) == counts.end())
            {
                counts[ngram] = 1;
            }
            else
            {
                counts[ngram]++;
            }
        }

        return counts;
    }

    cnn::real Precision(LossStats stats)
    {
        cnn::real prec = 1.0;
        for (int i = 0; i < NgramOrder; i++)
        {
            cnn::real x = stats[m_matchIndex + i] / (stats[m_hypIndex + i] + 0.001);
            prec *= pow(x, 1.0 / (cnn::real)NgramOrder);
        }
        return prec;
    }

};

namespace cnn {
    namespace metric {
        int levenshtein_distance(const std::vector<std::string> &s1, const std::vector<std::string> &s2);
    }
}