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
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates, 
            unsigned additional_input,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : s2tmodel(model, vocab_size_src, vocab_size_tgt, layers, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
        {
            swords = 0;
            twords = 0;
            nbr_turns = 0;
        }
    
        // return Expression of total loss
        // only has one pair of sentence so far
        virtual vector<Expression> build_graph(const Dialogue& cur_sentence, ComputationGraph& cg) = 0;

        // return Expression of total loss
        // only has one pair of sentence so far
        virtual vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) = 0;

        // return Expression of total loss
        // only has one pair of sentence so far
        virtual Expression build_graph(const TupleDialogue& cur_sentence, ComputationGraph& cg) = 0; 

        virtual Expression build_graph(const TupleDialogue& prv_sentence, const TupleDialogue& cur_sentence, ComputationGraph& cg) = 0;

        virtual std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.decode(source, cg, tdict);
        }

        virtual std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict)
        {
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

        // parallel decoding
        virtual vector<Sentence>batch_decode(const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict & tdict)
        {
            s2tmodel.reset();  /// reset network
            vector<Sentence> iret = s2tmodel.batch_decode(cur_sentence, cg, tdict);
            return iret;
        }

        virtual vector<Sentence> batch_decode(const vector<Sentence>& prv_sentence, const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict& tdict)
        {
            vector<Sentence> iret = s2tmodel.batch_decode(cur_sentence, cg, tdict);
            return iret;
        }

        std::vector<int> decode_tuple(const SentenceTuple &source, ComputationGraph& cg, cnn::Dict  &sdict, cnn::Dict  &tdict)
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.decode_tuple(source, cg, sdict, tdict);
        }

        std::vector<int> decode_tuple(const SentenceTuple &source, const SentenceTuple &cursource, ComputationGraph& cg, cnn::Dict  &sdict, cnn::Dict  &tdict)
        {
            return s2tmodel.decode_tuple(cursource, cg, sdict, tdict);
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

        void assign_cxt(ComputationGraph& cg, size_t nutt, vector<vector<vector<cnn::real>>>& v_cxt_s)
        {
            s2tmodel.assign_cxt(cg, nutt, v_cxt_s);
        }

        void serialise_cxt(ComputationGraph& cg)
        {
            s2tmodel.serialise_context(cg);
        }

        void reset()
        {
            twords = 0;
            swords = 0;
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
    private:
        vector<Expression> i_errs; 
    public:
        HREDModel(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const vector<SentencePair> & cur_sentence, ComputationGraph& cg) override
        {
            vector<Expression> object;
            vector<Sentence> insent, osent;
            i_errs.clear();
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2tmodel.reset();
            object = s2tmodel.build_graph(insent, osent, cg);

            Expression i_err = sum(object);
            s2txent = i_err;

            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);
            i_errs.push_back(i_err);
            return object;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const vector<SentencePair>& prv_sentence, const vector<SentencePair>& cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent;

            for (auto p : cur_sentence)
            {
                osent.push_back(p.first);
            }
            for (auto p : prv_sentence)
            {
                insent.push_back(p.second);
            }


            int nutt = cur_sentence.size();

            vector<Expression> object_prv_t2cur_s = s2tmodel.build_graph(insent, osent, cg);

            osent.clear(); insent.clear();
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            vector<Expression> object_cur_s2cur_t = s2tmodel.build_graph(insent, osent, cg);

            s2txent = s2txent + sum(object_cur_s2cur_t);

            Expression i_sum_err = sum(object_cur_s2cur_t) + sum(object_prv_t2cur_s);
            i_errs.push_back(i_sum_err);
            return object_cur_s2cur_t;
        }

        Expression build_graph(const vector<SentenceTuple> & cur_sentence, ComputationGraph& cg) override
        {
            Expression object;
            throw("not implemented");
            return object;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const vector<SentenceTuple>& prv_sentence, const vector<SentenceTuple>& cur_sentence, ComputationGraph& cg) override
        {
            Expression object;
            throw("not implemented");
            return object;
        }
    };


    /**
    Neural conversation model using sequence to sequence method
    arxiv.org/pdf/1506.05869v2.pdf
    */
    template <class DBuilder>
    class DialogueSeq2SeqModel : public DialogueProcessInfo<DBuilder> {
    private:
        vector<Expression> i_errs;

    public:
        DialogueSeq2SeqModel(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {
        }


        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const vector<SentencePair> & cur_sentence, ComputationGraph& cg) override
        {
            vector<Expression> object;
            vector<Sentence> insent, osent;

            i_errs.clear();

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2tmodel.reset();
            object = s2tmodel.build_graph(insent, osent, cg);

            s2txent = sum(object);

            i_errs.push_back(s2txent);
            return object;
        }

        /**
        concatenate the previous response and the current source as input, and predict the reponse of the current turn
        */
        vector<Expression> build_graph(const vector<SentencePair>& prv_sentence, const vector<SentencePair>& cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent;

            for (auto p : prv_sentence)
            {
                /// remove sentence ending
                Sentence i_s; 
                for (auto & w : p.second){
                    if (w != kSRC_EOS)
                        i_s.push_back(w);
                }
                insent.push_back(i_s);
            }

            size_t k = 0;
            for (auto p : cur_sentence)
            {
                /// remove sentence begining
                for (auto & w : p.first){
                    if (w != kSRC_SOS)
                        insent[k].push_back(w);
                }
                swords += insent[k].size() - 1;
                k++;
            }

            for (auto p : cur_sentence)
            {
                osent.push_back(p.second);

                twords += p.second.size() - 1;
            }

            int nutt = cur_sentence.size();

            vector<Expression> object_cur_s2cur_t = s2tmodel.build_graph(insent, osent, cg);
            Expression i_err = sum(object_cur_s2cur_t);

            i_errs.push_back(i_err);

            s2txent = s2txent + i_err;

            return object_cur_s2cur_t;
        }

        Expression build_graph(const vector<SentenceTuple> & cur_sentence, ComputationGraph& cg) override
        {
            Expression object;
            throw("not implemented");
            return object;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const vector<SentenceTuple>& prv_sentence, const vector<SentenceTuple>& cur_sentence, ComputationGraph& cg) override
        {
            Expression object;
            throw("not implemented");
            return object;
        }

        std::vector<int> decode(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict  &tdict) override
        {
            s2tmodel.reset();  /// reset network
            return s2tmodel.decode(source, cg, tdict);
        }

        std::vector<int> decode(const std::vector<int> &source, const std::vector<int>& cur, ComputationGraph& cg, cnn::Dict  &tdict) override
        {
            Sentence insent;

            /// remove sentence ending
            for (auto & w : source){
                if (w != kSRC_EOS)
                    insent.push_back(w);
            }

            /// remove sentence begining
            for (auto & w : cur){
                if (w != kSRC_SOS)
                    insent.push_back(w);
            }

            swords += insent.size() - 1;

            return s2tmodel.decode(insent, cg, tdict);
        }

        vector<Sentence>batch_decode(const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict & tdict)
        {
            s2tmodel.reset();  /// reset network
            vector<Sentence> iret = s2tmodel.batch_decode(cur_sentence, cg, tdict);
            return iret;
        }
        
        vector<Sentence> batch_decode(const vector<Sentence>& prv_sentence, const vector<Sentence>& cur_sentence, ComputationGraph& cg, cnn::Dict& tdict)
        {
            vector<Sentence> insent;

            for (auto p : prv_sentence)
            {
                /// remove sentence ending
                Sentence i_s;
                for (auto & w : p){
                    if (w != kSRC_EOS)
                        i_s.push_back(w);
                }
                insent.push_back(i_s);
            }

            size_t k = 0;
            for (auto p : cur_sentence)
            {
                /// remove sentence begining
                for (auto & w : p){
                    if (w != kSRC_SOS)
                        insent[k].push_back(w);
                }
                swords += insent[k].size() - 1;
                k++;
            }

            vector<Sentence> iret = s2tmodel.batch_decode(insent, cg, tdict);
            return iret;
        }

    };

    template <class DBuilder>
    class AttentionWithIntentionModel : public DialogueProcessInfo<DBuilder>{
        size_t align_dim;
    public:
        explicit AttentionWithIntentionModel(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {}

        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const Dialogue & cur_sentence, ComputationGraph& cg) override
        {
            vector<Expression> object;

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

            s2txent = sum(object);
            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return object;
        }

        /// for all speakers with history
        /// for feedforward network
        vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) override
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

            vector<Expression> s2terr = s2tmodel.build_graph(insent, osent, cg);
            Expression i_err = sum(s2terr);
            s2txent = s2txent + i_err;

            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return s2terr;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const TupleDialogue & cur_sentence, ComputationGraph& cg) override
        {
            Expression object;

            twords = 0;
            swords = 0;
            nbr_turns = 1;
            vector<Sentence> insent, osent, intention;
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.middle);
                intention.push_back(p.last);

                /// no recurrent
                twords += p.middle.size();  /// target doesn't have </s> so use the full observations
                swords += (p.first.size()> 0)?(p.first.size()-1):0;
            }

            s2tmodel.reset();
            s2tmodel.assign_cxt(cg, intention);
            vector<Expression> obj = s2tmodel.build_comp_graph(insent, osent, cg);
            if (obj.size() > 0)
            {
                object = sum(obj);

                s2txent = object;
                assert(twords == s2tmodel.tgt_words);
                assert(swords == s2tmodel.src_words);
            }
            return object;
        }

        /// for all speakers with history
        Expression build_graph(const TupleDialogue & prv_sentence, const TupleDialogue & cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent, intention;
            nbr_turns++;

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.middle);
                intention.push_back(p.last);

                /// no recurrent
                twords += p.middle.size();  /// target doesn't have </s> so use the full observations
                swords += (p.first.size()> 0) ? (p.first.size() - 1) : 0;
            }

            s2tmodel.assign_cxt(cg, intention);
            vector<Expression> obj = s2tmodel.build_comp_graph(insent, osent, cg);
            if (obj.size() > 0)
            {
                s2txent = sum(obj);
                assert(twords == s2tmodel.tgt_words);
                assert(swords == s2tmodel.src_words);
            }

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

    template <class DBuilder>
    class AttentionalConversation: public DialogueProcessInfo<DBuilder>{
        size_t align_dim;
    public:
        explicit AttentionalConversation(cnn::Model& model,
            const vector<unsigned int>& layers,
            unsigned vocab_size_src,
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned decoder_additional_input = 0,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : DialogueProcessInfo<DBuilder>(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, decoder_additional_input, mem_slots, iscale)
        {}

        // return Expression of total loss
        // only has one pair of sentence so far
        vector<Expression> build_graph(const Dialogue & cur_sentence, ComputationGraph& cg) override
        {
            vector<Expression> object;

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

            s2txent = sum(object);
            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return object;
        }

        /// for all speakers with history
        /// for feedforward network
        vector<Expression> build_graph(const Dialogue& prv_sentence, const Dialogue& cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent;
            nbr_turns++;
            twords = 0;
            swords = 0;

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.second);

                twords += p.second.size() - 1;
                swords += p.first.size() - 1;
            }

            s2tmodel.assign_cxt(cg, insent.size());
            vector<Expression> s2terr = s2tmodel.build_graph(insent, osent, cg);
            Expression i_err = sum(s2terr);
            s2txent = i_err;

            assert(twords == s2tmodel.tgt_words);
            assert(swords == s2tmodel.src_words);

            return s2terr;
        }

        // return Expression of total loss
        // only has one pair of sentence so far
        Expression build_graph(const TupleDialogue & cur_sentence, ComputationGraph& cg) override
        {
            Expression object;

            twords = 0;
            swords = 0;
            nbr_turns = 1;
            vector<Sentence> insent, osent, intention;
            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.middle);
                intention.push_back(p.last);

                /// no recurrent
                twords += p.middle.size();  /// target doesn't have </s> so use the full observations
                swords += (p.first.size()> 0) ? (p.first.size() - 1) : 0;
            }

            s2tmodel.reset();
            s2tmodel.assign_cxt(cg, intention);
            vector<Expression> obj = s2tmodel.build_comp_graph(insent, osent, cg);
            if (obj.size() > 0)
            {
                object = sum(obj);

                s2txent = object;
                assert(twords == s2tmodel.tgt_words);
                assert(swords == s2tmodel.src_words);
            }
            return object;
        }

        /// for all speakers with history
        Expression build_graph(const TupleDialogue & prv_sentence, const TupleDialogue & cur_sentence, ComputationGraph& cg) override
        {
            vector<Sentence> insent, osent, intention;
            nbr_turns++;

            for (auto p : cur_sentence)
            {
                insent.push_back(p.first);
                osent.push_back(p.middle);
                intention.push_back(p.last);

                /// no recurrent
                twords += p.middle.size();  /// target doesn't have </s> so use the full observations
                swords += (p.first.size()> 0) ? (p.first.size() - 1) : 0;
            }

            s2tmodel.assign_cxt(cg, intention);
            vector<Expression> obj = s2tmodel.build_comp_graph(insent, osent, cg);
            if (obj.size() > 0)
            {
                s2txent = sum(obj);
                assert(twords == s2tmodel.tgt_words);
                assert(swords == s2tmodel.src_words);
            }

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
            unsigned vocab_size_tgt,
            const vector<unsigned>& hidden_dim,
            unsigned hidden_replicates,
            unsigned additional_input,
            unsigned mem_slots = MEM_SIZE,
            cnn::real iscale = 1.0)
            : AttentionWithIntentionModel(model, layers, vocab_size_src, vocab_size_tgt, hidden_dim, hidden_replicates, additional_input, mem_slots, iscale)
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
