#pragma once

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/rnnem.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/expr-xtra.h"
#include "cnn/data-util.h"
#include "ext/dialogue/dialogue.h"
//#include "cxtencdec.h"
#include "cnn/math-util.h"
#include "ext/dialogue/attention_with_intention.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

using namespace std;

namespace cnn {

    template <class DBuilder>
    class DialogueProcessInfo{
    public:
        DBuilder s2tmodel;  /// source to target 

        int swords;
        int twords;
        int nbr_turns;

        Expression s2txent;

        DialogueProcessInfo(cnn::Model& model,
            unsigned layers,
            unsigned vocab_size_src,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates, 
            unsigned additional_input,
            unsigned mem_slots = MEM_SIZE,
            float iscale = 1.0)
            : s2tmodel(model, vocab_size_src, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
        {
            swords = 0;
            twords = 0;
            nbr_turns = 0;
        }
    
        // return Expression of total loss
        // only has one pair of sentence so far
        virtual Expression build_graph(const Dialogue& cur_sentence, ComputationGraph& cg) = 0;

        // return Expression of total loss
        // only has one pair of sentence so far
        virtual Expression build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) = 0;

#ifdef INPUT_UTF8
        virtual std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict<std::wstring> &tdict)
#else
        virtual std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
#endif
        {
            s2tmodel.reset();  /// reset network
//            s2tmodel.assign_tocxt(cg, 1);
            return s2tmodel.decode(source, cg, tdict);
        }

#ifdef INPUT_UTF8
        virtual std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict<std::wstring> &tdict)
#else
        virtual std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict)
#endif
        {
//            s2tmodel.assign_tocxt(cg, 1);
            return s2tmodel.decode(cur, cg, tdict);
        }

        virtual std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict,
            vector<vector<cnn::real>>& v_cxt_s, vector<vector<cnn::real>>& v_decoder_s)
        {
            s2tmodel.reset();  /// reset network
            vector<int> iret = s2tmodel.decode(source, cg, tdict);

            s2tmodel.serialise_context(cg, v_cxt_s, v_decoder_s);
            return iret;
        }

        virtual std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict,
            vector<vector<cnn::real>>& v_cxt_s, vector<vector<cnn::real>>& v_decoder_s)
        {
            if (v_cxt_s.size() > 0 && v_decoder_s.size() > 0)
                assign_cxt(cg, 1, v_cxt_s, v_decoder_s);
            vector<int> iret = s2tmodel.decode(cur, cg, tdict);

            s2tmodel.serialise_context(cg, v_cxt_s, v_decoder_s);
            return iret;
        }

        void assign_cxt(ComputationGraph& cg, size_t nutt)
        {
            twords = 0;
            swords = 0;
            s2tmodel.assign_cxt(cg, nutt);
        }

        void assign_cxt(ComputationGraph& cg, size_t nutt, vector<vector<cnn::real>>& v_cxt_s, vector<vector<cnn::real>>& v_decoder_s)
        {
            twords = 0;
            swords = 0;
            s2tmodel.assign_cxt(cg, nutt, v_cxt_s, v_decoder_s);
        }

        void serialise_cxt(ComputationGraph& cg)
        {
            s2tmodel.serialise_context(cg);
        }

        void reset()
        {
            s2tmodel.reset();
        }

        void collect_candidates(const std::vector<int>& response)
        {
            
        }

        void clear_candidates()
        {

        }

    public:

        /**
        @bcharlevel : true if output in character level, so not insert blank symbol after each output. default false.
        */
#ifdef INPUT_UTF8
        wstring respond(Model &model, wstring strquery, Dict<std::wstring>& td, bool bcharlevel = false)
#else
        string respond(Model &model, string strquery, Dict & td, 
            vector<vector<cnn::real>>& last_cxt_s,
            vector<vector<cnn::real>>& last_decoder_s,
            bool bcharlevel = false)
#endif
        {
#ifdef INPUT_UTF8
            wstring shuman;
            wstring response;
#else
            string shuman;
            string response;
#endif
            unsigned lines = 0;

            vector<int> decode_output;
            vector<int> shuman_input;

            shuman = "<s> " + strquery + " </s>";

            convertHumanQuery(shuman, shuman_input, td);

            ComputationGraph cg;
            if (prv_response.size() == 0)
                decode_output = decode(shuman_input, cg, td, last_cxt_s, last_decoder_s);
            else
                decode_output = decode(prv_response, shuman_input, cg, td, last_cxt_s, last_decoder_s);

            if (verbose)
            {
#ifdef INPUT_UTF8
                wcout << L"Agent: ";
                response = L"";
#else
                cout << "Agent: ";
                response = "";
#endif
            }

            for (auto pp : decode_output)
            {
                if (verbose)
                {
#ifdef INPUT_UTF8
                    wcout << td.Convert(pp) << L" ";
#else
                    if (!bcharlevel)
                        cout << td.Convert(pp) << " ";
                    else
                        cout << td.Convert(pp);
#endif
                }

                if (pp != kSRC_EOS && pp != kSRC_SOS)
                    response = response + td.Convert(pp);

                if (verbose)
                {
                    if (!bcharlevel)
#ifdef INPUT_UTF8
                        wcout << L" ";
#else
                        cout << " ";
#endif
                }
            }

            if (verbose)
                cout << endl;

            prv_response = decode_output;
            return response; 
        }

        /// return levenshtein between responses and reference
        int respond(vector<SentencePair> diag, Dict & td, bool bcharlevel = false)
        {
            string shuman;
            string response;

            unsigned lines = 0;

            int iDist = 0;

            vector<int> decode_output;
            vector<int> shuman_input;

            prv_response.clear();

            ComputationGraph cg;

            for (auto p : diag)
            {
                if (prv_response.size() == 0)
                    decode_output = decode(p.first, cg, td);
                else
                    decode_output = decode(prv_response, p.first, cg, td);
                cout << "user : ";
                for (auto pp : p.first)
                {
                    if (!bcharlevel)
                        cout << td.Convert(pp) << " ";
                    else 
                        cout << td.Convert(pp);
                }
                cout << endl;

                cout << "Agent: ";
                for (auto pp : decode_output)
                {
                    if (!bcharlevel)
                        cout << td.Convert(pp) << " ";
                    else
                        cout << td.Convert(pp);
                }
                cout << endl;

                prv_response = decode_output;

                /// compute distance
                vector<string> sref;
                for (auto pp : p.second)
                    sref.push_back(td.Convert(pp)); 
                vector<string>sres;
                for (auto pp : decode_output)
                    sres.push_back(td.Convert(pp));

                iDist += cnn::math::levenshtein_distance(sref, sres);
            }
            return iDist;
        }

        /// return levenshtein between responses and reference
        int respond(vector<SentencePair> diag, vector<SentencePair>& results, Dict & td)
        {
            string shuman;
            string response;

            unsigned lines = 0;

            int iDist = 0;

            vector<int> decode_output;
            vector<int> shuman_input;

            prv_response.clear();

            ComputationGraph cg;

            results.clear();
            for (auto p : diag)
            {
                SentencePair input_response; 

                if (prv_response.size() == 0)
                    decode_output = decode(p.first, cg, td);
                else
                    decode_output = decode(prv_response, p.first, cg, td);
                cout << "user : ";
                for (auto pp : p.first)
                {
                    cout << td.Convert(pp) << " ";
                }
                cout << endl;

                cout << "Agent: ";
                for (auto pp : decode_output)
                {
                    cout << td.Convert(pp) << " ";
                }
                cout << endl;

                prv_response = decode_output;

                /// compute distance
                vector<string> sref;
                for (auto pp : p.second)
                    sref.push_back(td.Convert(pp));
                vector<string>sres;
                for (auto pp : decode_output)
                    sres.push_back(td.Convert(pp));

                input_response = make_pair(p.first, decode_output);
                results.push_back(input_response);

                iDist += cnn::math::levenshtein_distance(sref, sres);
            }
            return iDist;
        }
    };

    /**
    HRED model for dialogue modeling
    http://arxiv.org/pdf/1507.04808v1.pdf
    */
    template <class DBuilder>
    class HREDModel : public DialogueProcessInfo<DBuilder>{
    public:
        HREDModel(cnn::Model& model,
            unsigned layers,
            unsigned vocab_size_src,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            float iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const vector<SentencePair> & cur_sentence, ComputationGraph& cg) override
        {
            Expression object;
            vector<Sentence> insent, osent;
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            object = s2tmodel.build_graph(insent, osent, cg);

            s2txent = object;

            return object;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const vector<SentencePair>& prv_sentence, const vector<SentencePair>& cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent;
            twords = swords = 0;

            for (auto p : cur_sentence)
            {
                osent.push_back(p.first);
            }
            for (auto p : prv_sentence)
            {
                insent.push_back(p.second);
            }


            int nutt = cur_sentence.size();

            Expression object_prv_t2cur_s;
                //= s2tmodel.build_graph_target_source(insent, osent, cg);

            osent.clear(); insent.clear();
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            Expression object_cur_s2cur_t = s2tmodel.build_graph(insent, osent, cg);

            s2txent = object_cur_s2cur_t;
            return object_cur_s2cur_t + object_prv_t2cur_s;
        }
    };


    /**
    Neural conversation model using sequence to sequence method
    arxiv.org/pdf/1506.05869v2.pdf
    */
    template <class DBuilder>
    class DialogueSeq2SeqModel : public DialogueProcessInfo<DBuilder> {

    public:
        DialogueSeq2SeqModel(cnn::Model& model,
            unsigned layers,
            unsigned vocab_size_src,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, hidden_dim, hidden_replicates)
        {
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const vector<SentencePair> & cur_sentence, ComputationGraph& cg) override
        {
            Expression object;
            vector<Sentence> insent, osent;
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            object = s2tmodel.build_graph(insent, osent, cg);

            s2txent = object;

            return object;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const vector<SentencePair>& prv_sentence, const vector<SentencePair>& cur_sentence, ComputationGraph& cg) override
        {
            return build_graph(cur_sentence, cg);
        }
    };

    template <class DBuilder>
    class AttentionWithIntentionModel : public DialogueProcessInfo<DBuilder>{
        size_t align_dim;
    public:
        explicit AttentionWithIntentionModel(cnn::Model& model,
            unsigned layers,
            unsigned vocab_size_src,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            float iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {}

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const Dialogue & cur_sentence, ComputationGraph& cg) override
        {
            Expression object;

            twords = 0;
            swords = 0;
            nbr_turns = 1;
            vector<Sentence> insent, osent;
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2tmodel.reset();
            object = s2tmodel.build_graph(insent, osent, cg);

            s2txent = object;
            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return object;
        }

        /// for all speakers with history
        Expression build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent;
            nbr_turns ++;

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2txent = s2tmodel.build_graph(insent, osent, cg);

            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return s2txent;
        }

        void clear()
        {
            s2tmodel.clear();
        }

        // return Expression of total loss
        // only has one pair of sentence so far
#ifdef INPUT_UTF8
        vector<int> decode(const Sentence& source, ComputationGraph& cg, Dict<std::wstring>&  td)
#else
        vector<int> decode(const Sentence& source, ComputationGraph& cg, Dict &  td)
#endif
        {
            vector<int> results;

            s2tmodel.reset();

            results = s2tmodel.decode(source, cg, td);
            twords = results.size() - 1;
            swords = source.size() - 1;

            serialise_cxt(cg);

            return results;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
#ifdef INPUT_UTF8
        vector<int> decode(const Sentence& prv_response, const Sentence& cur_source, ComputationGraph& cg, Dict<std::wstring>&  td)
#else
        vector<int> decode(const Sentence& prv_response, const Sentence& cur_source, ComputationGraph& cg, Dict &  td)
#endif
        {
//            s2tmodel.assign_tocxt(cg, 1);
            vector<int> results
                = s2tmodel.decode(cur_source, cg, td);
            twords = results.size() - 1;
            swords = cur_source.size() - 1;

            serialise_cxt(cg);
            return results;
        }
    };

    /** 
    reinforcement learning for AWI model
    */
    template <class DBuilder>
    class RLAttentionWithIntentionModel : public AttentionWithIntentionModel<DBuilder>{
    public:
        RLAttentionWithIntentionModel(cnn::Model& model,
            unsigned layers,
            unsigned vocab_size_src,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned additional_input,
            unsigned mem_slots = MEM_SIZE,
            float iscale = 1.0)
            : AttentionWithIntentionModel(model, layers, vocab_size_src, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
        {
        }

        Expression rl_build_graph(ComputationGraph& cg)
        {
            Expression object;

            object = s2tmodel.rl_build_graph(cg);

            return object;
        }

    };

}; // namespace cnn
