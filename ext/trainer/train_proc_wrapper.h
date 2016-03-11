#pragma once

#include "ext/trainer/train_proc.h"

template<class rnn_t, class TrainProc>
void prt_model_info(size_t LAYERS, size_t VOCAB_SIZE_SRC, const vector<unsigned>& dims, size_t nreplicate, size_t decoder_additiona_input_to, size_t mem_slots, cnn::real scale)
{
    cerr << "layer = " << LAYERS << endl;
    cerr << "vocab size = " << VOCAB_SIZE_SRC << endl;
    cerr << "dims = ";
    for (auto & p : dims)
    {
        cerr << " " << p;
    }
    cerr << endl;
    cerr << "nreplicate = " << nreplicate << endl;
    cerr << "decoder_additional_input_to = " << decoder_additiona_input_to << endl;
    cerr << "mem_slots = " << mem_slots << endl;
    cerr << "scale = " << scale << endl;
}

template<class rnn_t, class TrainProc>
Trainer* select_trainer(variables_map vm, Model* model)
{
    Trainer* sgd = nullptr;
    if (vm["trainer"].as<string>() == "momentum")
    {
        if (vm.count("mom") > 0)
            sgd = new MomentumSGDTrainer(model, 1e-6, vm["eta"].as<cnn::real>(), vm["mom"].as<cnn::real>());
        else
            sgd = new MomentumSGDTrainer(model, 1e-6, vm["eta"].as<cnn::real>());
    }
    if (vm["trainer"].as<string>() == "sgd")
        sgd = new SimpleSGDTrainer(model, 1e-6, vm["eta"].as<cnn::real>());
    if (vm["trainer"].as<string>() == "adagrad")
        sgd = new AdagradTrainer(model, 1e-6, vm["eta"].as<cnn::real>());
    if (vm["trainer"].as<string>() == "adadelta")
        sgd = new AdadeltaTrainer(model, 1e-6, vm["eta"].as<cnn::real>());
    if (vm["trainer"].as<string>() == "rmsprop")
        sgd = new RmsPropTrainer(model, 1e-6, vm["eta"].as<cnn::real>());
    if (vm["trainer"].as<string>() == "rmspropwithmomentum")
        sgd = new RmsPropWithMomentumTrainer(model, 1e-6, vm["eta"].as<cnn::real>());
    sgd->clip_threshold = vm["clip"].as<cnn::real>();
    sgd->eta_decay = vm["eta_decay"].as<cnn::real>();

    return sgd;
}

vector<bool> get_padding_position(variables_map vm)
{
    vector<bool> padding_position;
    if (vm["padding"].as<bool>())
    {
        padding_position.push_back(vm["padding_source_to_back"].as<bool>());
        padding_position.push_back(vm["padding_target_to_back"].as<bool>());
    }
    return padding_position;
}

template <class rnn_t, class TrainProc>
int main_body(variables_map vm, size_t nreplicate = 0, size_t decoder_additiona_input_to = 0, size_t mem_slots = MEM_SIZE)
{
#ifdef INPUT_UTF8
    kSRC_SOS = sd.Convert(L"<s>");
    kSRC_EOS = sd.Convert(L"</s>");
#else
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
#endif
    verbose = vm.count("verbose");
    g_train_on_turns = vm["turns"].as<int>();

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel, testcorpus;
    string line;
    cnn::real largest_dev_cost = 9e+99;
    TrainProc  * ptrTrainer = nullptr;

    if (vm.count("readdict"))
    {
        string fname = vm["readdict"].as<string>();
#ifdef INPUT_UTF8
        wifstream in(fname, wifstream::in);
        boost::archive::text_wiarchive ia(in);
#else
        ifstream in(fname, ifstream::in);
        boost::archive::text_iarchive ia(in);
#endif
        if (!in.is_open())
            throw("cannot open " + fname);
        ia >> sd;
        sd.Freeze();
    }

    if ((vm.count("train") > 0 && vm["epochsize"].as<int>() == -1) || vm.count("writedict") > 0 || vm.count("train-lda") > 0)
    {
        cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
        training = read_corpus(vm["train"].as<string>(), sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), false,
            vm.count("charlevel") > 0);
        sd.Freeze(); // no new word types allowed

        training_numturn2did = get_numturn2dialid(training);

        if (vm.count("writedict"))
        {
            string fname = vm["writedict"].as<string>();
#ifdef INPUT_UTF8
            wstring wfname;
            wfname.assign(fname.begin(), fname.end());
            wofstream ofs(wfname);
            boost::archive::text_woarchive oa(ofs);
#else
            ofstream on(fname);
            boost::archive::text_oarchive oa(on);
#endif
            oa << sd;
        }
    }
    else
    {
        if (vm.count("readdict") == 0)
        {
            throw std::invalid_argument("must have either training corpus or dictionary");
        }
    }

    LAYERS = vm["layers"].as<int>();
    HIDDEN_DIM = vm["hidden"].as<int>();
    ALIGN_DIM = vm["align"].as<int>();

    string flavour = builder_flavour(vm);
    VOCAB_SIZE_SRC = sd.size();
    VOCAB_SIZE_TGT = sd.size(); /// use the same dictionary
    nparallel = vm["nparallel"].as<int>();
    mbsize = vm["mbsize"].as < int >();

    if (vm.count("beamsearchdecode"))
    {
        beam_search_decode = vm["beamsearchdecode"].as<int>();
    }

    if (vm.count("devel")) {
        cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
        devel = read_corpus(vm["devel"].as<string>(), sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), true, vm.count("charlevel") > 0);
        devel_numturn2did = get_numturn2dialid(devel);
    }

    if (vm.count("testcorpus")) {
        cerr << "Reading test corpus from " << vm["testcorpus"].as<string>() << "...\n";
        testcorpus = read_corpus(vm["testcorpus"].as<string>(), sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), true);
        test_numturn2did = get_numturn2dialid(testcorpus);
        if (vm.count("outputfile") == 0)
        {
            cerr << "missing --outputfile" << endl;
            throw std::invalid_argument("missing --outputfile");
        }
    }

    if (vm.count("train-lda") > 0)
    {
        ptrTrainer->lda_train(vm, training, devel, sd);
    }

    string fname;
    if (vm.count("parameters")) {
        fname = vm["parameters"].as<string>();
    }
    else {
        ostringstream os;
        os << "attentionwithintention"
            << '_' << LAYERS
            << '_' << HIDDEN_DIM
            << '_' << flavour
            << "-pid" << getpid() << ".params";
        fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    Trainer* sgd = select_trainer<rnn_t, TrainProc>(vm, &model);

    cerr << "%% Using " << flavour << " recurrent units" << endl;

    std::vector<unsigned> dims;
    dims.resize(4);
    if (!vm.count("hidden"))
        dims[ENCODER_LAYER] = HIDDEN_DIM;
    else
        dims[ENCODER_LAYER] = (unsigned)vm["hidden"].as<int>();
    dims[DECODER_LAYER] = dims[ENCODER_LAYER]; /// if not specified, encoder and decoder have the same dimension

    if (!vm.count("align"))
        dims[ALIGN_LAYER] = ALIGN_DIM;
    else
        dims[ALIGN_LAYER] = (unsigned)vm["align"].as<int>();
    if (!vm.count("intentiondim"))
        dims[INTENTION_LAYER] = HIDDEN_DIM;
    else
        dims[INTENTION_LAYER] = (unsigned)vm["intentiondim"].as<int>();


    std::vector<unsigned int> layers;
    layers.resize(4, LAYERS);
    if (!vm.count("intentionlayers"))
        layers[INTENTION_LAYER] = vm["intentionlayers"].as<size_t>();
    rnn_t hred(model, layers, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, mem_slots, vm["scale"].as<cnn::real>());
    prt_model_info<rnn_t, TrainProc>(LAYERS, VOCAB_SIZE_SRC, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, mem_slots, vm["scale"].as<cnn::real>());

    /// read word class information
    if (vm["word2clsfn"].as<string>().size() > 0 && vm["clssizefn"].as<string>().size() > 0)
    {
        hred.load_cls_info_from_file(vm["word2clsfn"].as<string>(), vm["clssizefn"].as<string>(), sd, model);
    }

    /// read embedding if specified
    if (vm.count("embeddingfn") > 0)
    {
        map<int, vector<cnn::real>> vWordEmbedding;
        string emb_filename = vm["embeddingfn"].as<string>();
        read_embedding(emb_filename, sd, vWordEmbedding);
        hred.init_word_embedding(vWordEmbedding);
        if (vm.count("dumpembeddingfn") > 0)
            hred.dump_word_embedding(vWordEmbedding, sd, vm["dumpembeddingfn"].as<string>());
    }

    if (vm.count("initialise"))
    {
        string fname = vm["initialise"].as<string>();
        ifstream in(fname, ifstream::in);
        if (in.is_open())
        {
            boost::archive::text_iarchive ia(in);
            ia >> model;
        }
    }

    ptrTrainer = new TrainProc();

    if (vm["pretrain"].as<cnn::real>() > 0)
    {
        ptrTrainer->supervised_pretrain(model, hred, training, devel, *sgd, fname, vm["pretrain"].as<cnn::real>(), 1);
        delete sgd;

        /// reopen sgd
        sgd = select_trainer<rnn_t, TrainProc>(vm, &model);
    }

    if (vm.count("sampleresponses"))
    {
        cerr << "Reading sample corpus from " << vm["sampleresponses"].as<string>() << "...\n";
        training = read_corpus(vm["sampleresponses"].as<string>(), sd, kSRC_SOS, kSRC_EOS);
        ptrTrainer->collect_sample_responses(hred, training);
    }
    if (vm.count("dialogue"))
    {
        if (vm.count("outputfile") == 0)
        {
            throw std::invalid_argument("missing recognition output file");
        }
        ptrTrainer->dialogue(model, hred, vm["outputfile"].as<string>(), sd);
    }
    if (vm.count("reinforce") && vm.count("nparallel") && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        // a mirrow of the agent to generate decoding results so that their results can be evaluated
        // this is not efficient implementation, better way is to share model parameters
        int n_reinforce_train = vm["num_reinforce_train"].as<int>();
        cnn::real largest_cost = 9e+99;
        ptrTrainer->reset_smoothed_ppl();
        for (size_t k_reinforce = 0; k_reinforce <= n_reinforce_train; k_reinforce++)
        {
            Model model_mirrow;
            string fname;
            if (vm.count("parameters") > 0 && k_reinforce == 0) {
                fname = vm["parameters"].as<string>();

                ofstream out(fname, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else if (vm.count("initialise") > 0){
                fname = vm["initialise"].as<string>();
            }
            else
                throw("need to specify either parameters or initialise model file name");
            rnn_t hred_agent_mirrow(model_mirrow, layers, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, mem_slots, vm["scale"].as<cnn::real>());
            ifstream in(fname, ifstream::in);
            if (in.is_open())
            {
                boost::archive::text_iarchive ia(in);
                ia >> model_mirrow;
            }

            cnn::real threshold_prob;
            threshold_prob = 1.0 - k_reinforce / (vm["num_reinforce_train"].as<int>() + 0.0);

            size_t each_epoch = min<int>(2, vm["epochs"].as<int>() / n_reinforce_train);
            ptrTrainer->REINFORCEtrain(model, hred, hred_agent_mirrow, training, devel, *sgd, fname, sd, each_epoch * n_reinforce_train, vm["nparallel"].as<int>(), largest_cost, vm["reward_baseline"].as<cnn::real>(), threshold_prob);
        }
    }
    else if (vm["epochsize"].as<int>() >1 && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {   // split data into nparts and train
        training.clear();
        ptrTrainer->split_data_batch_train(vm["train"].as<string>(), model, hred, devel, *sgd, fname, vm["epochs"].as<int>(), vm["nparallel"].as<int>(), vm["epochsize"].as<int>(), vm["segmental_training"].as<bool>(), vm["do_gradient_check"].as<bool>(),
            get_padding_position(vm));
    }
    else if (vm.count("nparallel") && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        ptrTrainer->batch_train(model, hred, training, devel, *sgd, fname, vm["epochs"].as<int>(), vm["nparallel"].as<int>(), largest_dev_cost, vm["segmental_training"].as<bool>(), true, vm["do_gradient_check"].as<bool>(), true, get_padding_position(vm), kSRC_EOS);
    }
    else if (!vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        ptrTrainer->train(model, hred, training, devel, *sgd, fname, vm["epochs"].as<int>(), vm.count("charlevel") > 0, vm.count("nosplitdialogue"));
    }
    else if (vm.count("testcorpus"))
    {
        if (vm.count("outputfile") == 0)
        {
            throw std::invalid_argument("missing recognition output file");
        }
        ptrTrainer->test(model, hred, testcorpus, vm["outputfile"].as<string>(), sd, test_numturn2did, vm["segmental_training"].as<bool>());
    }

    delete sgd;
    delete ptrTrainer;

    return EXIT_SUCCESS;
}

/**
The higher level training body for classification tasks

Such task reads a sequence of input, does feature representation of the input, and outputs a label. 
This task includes speech recognition, image recognition, OCR, CTC, etc. 
This training inherets a training process, TrainProcess, for generation. 

The training can be made efficient using data parallel. Because the output response is one for each input streams,
the training can use one softmax operation that can work on multiple inputs.

The data is read with a format such as <sentences> ||| id 
*/
template <class rnn_t, class TrainProc>
int classification_main_body(variables_map vm, size_t nreplicate = 0, size_t decoder_additiona_input_to = 0)
{
#ifdef INPUT_UTF8
    kSRC_SOS = sd.Convert(L"<s>");
    kSRC_EOS = sd.Convert(L"</s>");
#else
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
#endif
    verbose = vm.count("verbose");

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel, testcorpus;
    string line;
    cnn::real largest_dev_cost = 9e+99;
    TrainProc  * ptrTrainer = nullptr;
    pair<int, int> dialogue_colums = make_pair<int, int>(2, 4);

    if (vm.count("readdict"))
    {
        string fname = vm["readdict"].as<string>();
#ifdef INPUT_UTF8
        wifstream in(fname, wifstream::in);
        boost::archive::text_wiarchive ia(in);
#else
        ifstream in(fname, ifstream::in);
        boost::archive::text_iarchive ia(in);
#endif
        if (!in.is_open())
            throw("cannot open " + fname);
        ia >> sd;
        sd.Freeze();
    }

    string id2strfname = vm["extract_id_and_string"].as<string>();
#ifdef INPUT_UTF8
    wifstream in(id2strfname , wifstream::in);
    boost::archive::text_wiarchive ia(in);
#else
    ifstream in(id2strfname, ifstream::in);
    boost::archive::text_iarchive ia(in);
#endif
    if (!in.is_open())
        throw("cannot open " + id2strfname);
    ia >> id2str;
    id2str.Freeze();
    
    if ((vm.count("train") > 0 && vm["epochsize"].as<int>() == -1) || vm.count("writedict") > 0 || vm.count("train-lda") > 0)
    {
        cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
        training = read_corpus(vm["train"].as<string>(), sd, kSRC_SOS, kSRC_EOS, false, dialogue_colums);
        sd.Freeze(); // no new word types allowed

        training_numturn2did = get_numturn2dialid(training);

        if (vm.count("writedict"))
        {
            string fname = vm["writedict"].as<string>();
#ifdef INPUT_UTF8
            wstring wfname;
            wfname.assign(fname.begin(), fname.end());
            wofstream ofs(wfname);
            boost::archive::text_woarchive oa(ofs);
#else
            ofstream on(fname);
            boost::archive::text_oarchive oa(on);
#endif
            oa << sd;
        }
    }
    else
    {
        if (vm.count("readdict") == 0)
        {
            throw std::invalid_argument("must have either training corpus or dictionary");
        }
    }

    LAYERS = vm["layers"].as<int>();
    HIDDEN_DIM = vm["hidden"].as<int>();
    ALIGN_DIM = vm["align"].as<int>();

    string flavour = builder_flavour(vm);
    VOCAB_SIZE_SRC = sd.size();
    VOCAB_SIZE_TGT = id2str.size(); 
    nparallel = vm["nparallel"].as<int>();
    mbsize = vm["mbsize"].as < int >();

    if (vm.count("beamsearchdecode"))
    {
        beam_search_decode = vm["beamsearchdecode"].as<int>();
    }

    if (vm.count("devel")) {
        cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
        devel = read_corpus(vm["devel"].as<string>(), sd, kSRC_SOS, kSRC_EOS, true, dialogue_colums);
        devel_numturn2did = get_numturn2dialid(devel);
    }

    if (vm.count("testcorpus")) {
        cerr << "Reading test corpus from " << vm["testcorpus"].as<string>() << "...\n";
        testcorpus = read_corpus(vm["testcorpus"].as<string>(), sd, kSRC_SOS, kSRC_EOS, true, dialogue_colums);
        test_numturn2did = get_numturn2dialid(testcorpus);
    }

    string fname;
    if (vm.count("parameters")) {
        fname = vm["parameters"].as<string>();
    }
    else {
        ostringstream os;
        os << "attentionwithintention"
            << '_' << LAYERS
            << '_' << HIDDEN_DIM
            << '_' << flavour
            << "-pid" << getpid() << ".params";
        fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    Trainer* sgd = select_trainer<rnn_t, TrainProc>(vm, &model);

    cerr << "%% Using " << flavour << " recurrent units" << endl;

    std::vector<unsigned> dims;
    dims.resize(4);
    if (!vm.count("hidden"))
        dims[ENCODER_LAYER] = HIDDEN_DIM;
    else
        dims[ENCODER_LAYER] = (unsigned)vm["hidden"].as<int>();
    dims[DECODER_LAYER] = dims[ENCODER_LAYER]; /// if not specified, encoder and decoder have the same dimension

    if (!vm.count("align"))
        dims[ALIGN_LAYER] = ALIGN_DIM;
    else
        dims[ALIGN_LAYER] = (unsigned)vm["align"].as<int>();
    if (!vm.count("intentiondim"))
        dims[INTENTION_LAYER] = HIDDEN_DIM;
    else
        dims[INTENTION_LAYER] = (unsigned)vm["intentiondim"].as<int>();


    std::vector<unsigned int> layers;
    layers.resize(4, LAYERS);
    if (!vm.count("intentionlayers"))
        layers[INTENTION_LAYER] = vm["intentionlayers"].as<size_t>();
    rnn_t hred(model, layers, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, 0, vm["scale"].as<cnn::real>());
    prt_model_info<rnn_t, TrainProc>(LAYERS, VOCAB_SIZE_SRC, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, 0, vm["scale"].as<cnn::real>());

    /// read embedding if specified
    if (vm["embeddingfn"].as<string>().size() > 0)
    {
        map<int, vector<cnn::real>> vWordEmbedding;
        string emb_filename = vm["embeddingfn"].as<string>();
        read_embedding(emb_filename, sd, vWordEmbedding);
        hred.init_word_embedding(vWordEmbedding);
        if (vm.count("dumpembeddingfn") > 0)
            hred.dump_word_embedding(vWordEmbedding, sd, vm["dumpembeddingfn"].as<string>());
    }

    if (vm.count("initialise"))
    {
        string fname = vm["initialise"].as<string>();
        ifstream in(fname, ifstream::in);
        if (in.is_open())
        {
            boost::archive::text_iarchive ia(in);
            ia >> model;
        }
    }

    ptrTrainer = new TrainProc();

    if (vm.count("dialogue"))
    {
        if (vm.count("outputfile") == 0)
        {
            throw std::invalid_argument("missing recognition output file");
        }
        ptrTrainer->dialogue(model, hred, vm["outputfile"].as<string>(), sd);
    }
    if (vm.count("reinforce") && vm.count("nparallel") && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        // a mirrow of the agent to generate decoding results so that their results can be evaluated
        // this is not efficient implementation, better way is to share model parameters
        int n_reinforce_train = vm["num_reinforce_train"].as<int>();
        cnn::real largest_cost = 9e+99;
        ptrTrainer->reset_smoothed_ppl();
        for (size_t k_reinforce = 0; k_reinforce <= n_reinforce_train; k_reinforce++)
        {
            Model model_mirrow;
            string fname;
            if (vm.count("parameters") > 0 && k_reinforce == 0) {
                fname = vm["parameters"].as<string>();

                ofstream out(fname, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else if (vm.count("initialise") > 0){
                fname = vm["initialise"].as<string>();
            }
            else
                throw("need to specify either parameters or initialise model file name");
            rnn_t hred_agent_mirrow(model_mirrow, layers, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, 0, vm["scale"].as<cnn::real>());
            ifstream in(fname, ifstream::in);
            if (in.is_open())
            {
                boost::archive::text_iarchive ia(in);
                ia >> model_mirrow;
            }

            cnn::real threshold_prob;
            threshold_prob = 1.0 - k_reinforce / (vm["num_reinforce_train"].as<int>() + 0.0);

            size_t each_epoch = min<int>(2, vm["epochs"].as<int>() / n_reinforce_train);
            ptrTrainer->REINFORCEtrain(model, hred, hred_agent_mirrow, training, devel, *sgd, fname, sd, each_epoch * n_reinforce_train, vm["nparallel"].as<int>(), largest_cost, vm["reward_baseline"].as<cnn::real>(), threshold_prob);
        }
    }
    else if (vm["epochsize"].as<int>() >1 && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {   // split data into nparts and train
        training.clear();
        ptrTrainer->split_data_batch_train(vm["train"].as<string>(), model, hred, devel, *sgd, fname, vm["epochs"].as<int>(), vm["nparallel"].as<int>(), vm["epochsize"].as<int>(), vm["segmental_training"].as<bool>(), vm["do_gradient_check"].as<bool>());
    }
    else if (vm.count("nparallel") && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        ptrTrainer->batch_train(model, hred, training, devel, *sgd, fname, vm["epochs"].as<int>(), vm["nparallel"].as<int>(), largest_dev_cost, vm["segmental_training"].as<bool>(), true, vm["do_gradient_check"].as<bool>(), true);
    }
    else if (vm.count("testcorpus"))
    {
        if (vm.count("outputfile") == 0)
        {
            throw std::invalid_argument("missing recognition output file");
        }
        ptrTrainer->test(model, hred, testcorpus, vm["outputfile"].as<string>(), sd, test_numturn2did);
    }

    delete sgd;
    delete ptrTrainer;

    return EXIT_SUCCESS;
}

template <class TrainProc>
int clustering_main_body(variables_map vm)
{
#ifdef INPUT_UTF8
    kSRC_SOS = sd.Convert(L"<s>");
    kSRC_EOS = sd.Convert(L"</s>");
#else
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
#endif
    verbose = vm.count("verbose");

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel, testcorpus;
    string line;
    cnn::real largest_dev_cost = 9e+99;
    TrainProc  * ptrTrainer = nullptr;

    if (vm.count("readdict"))
    {
        string fname = vm["readdict"].as<string>();
#ifdef INPUT_UTF8
        wifstream in(fname, wifstream::in);
        boost::archive::text_wiarchive ia(in);
#else
        ifstream in(fname, ifstream::in);
        boost::archive::text_iarchive ia(in);
#endif
        if (!in.is_open())
            throw("cannot open " + fname);
        ia >> sd;
        sd.Freeze();
    }

    if ((vm.count("train") > 0 && vm["epochsize"].as<int>() == -1) || vm.count("writedict") > 0 || vm.count("train-lda") > 0)
    {
        cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
        training = read_corpus(vm["train"].as<string>(), sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), false,
            vm.count("charlevel") > 0);
        sd.Freeze(); // no new word types allowed

        training_numturn2did = get_numturn2dialid(training);

        if (vm.count("writedict"))
        {
            string fname = vm["writedict"].as<string>();
#ifdef INPUT_UTF8
            wstring wfname;
            wfname.assign(fname.begin(), fname.end());
            wofstream ofs(wfname);
            boost::archive::text_woarchive oa(ofs);
#else
            ofstream on(fname);
            boost::archive::text_oarchive oa(on);
#endif
            oa << sd;
        }
    }
    else
    {
        if (vm.count("readdict") == 0)
        {
            throw std::invalid_argument("must have either training corpus or dictionary");
        }
    }

    if (vm.count("devel")) {
        cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
        devel = read_corpus(vm["devel"].as<string>(), sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), true, vm.count("charlevel") > 0);
        devel_numturn2did = get_numturn2dialid(devel);
    }

    if (vm.count("testcorpus")) {
        cerr << "Reading test corpus from " << vm["testcorpus"].as<string>() << "...\n";
        testcorpus = read_corpus(vm["testcorpus"].as<string>(), sd, kSRC_SOS, kSRC_EOS, vm["mbsize"].as<int>(), true);
        test_numturn2did = get_numturn2dialid(testcorpus);
    }

    if (vm.count("train-lda") > 0)
    {
        ptrTrainer->lda_train(vm, training, devel, sd);
    }

    if (vm.count("test-lda") > 0)
    {
        ptrTrainer->lda_test(vm, devel, sd);
    }

    if (vm.count("ngram-training") > 0)
    {
        ptrTrainer->ngram_train(vm, training, sd);
    }

    if (vm.count("ngram-clustering") > 0)
    {
        ptrTrainer->ngram_clustering(vm, training, sd);
    }

    if (vm.count("ngram_one_pass_clustering") > 0)
        ptrTrainer->ngram_one_pass_clustering(vm, training, sd);

    delete ptrTrainer;

    return EXIT_SUCCESS;
}

template <class TrainProc>
int hirearchical_clustering_main_body(variables_map vm)
{
#ifdef INPUT_UTF8
    kSRC_SOS = sd.Convert(L"<s>");
    kSRC_EOS = sd.Convert(L"</s>");
#else
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
#endif
    verbose = vm.count("verbose");

    CorpusWithClassId training;
    string line;
    cnn::real largest_dev_cost = 9e+99;
    TrainProc  * ptrTrainer = nullptr;

    if (vm.count("extract_id_and_string") > 0)
    {
        get_string_and_its_id(vm["train"].as<string>(), make_pair<int, int>(5,4), vm["extract_id_and_string"].as<string>());
        return EXIT_SUCCESS;
    }

    if (vm.count("readdict") > 0)
    {
        string fname = vm["readdict"].as<string>();
#ifdef INPUT_UTF8
        wifstream in(fname, wifstream::in);
        boost::archive::text_wiarchive ia(in);
#else
        ifstream in(fname, ifstream::in);
        boost::archive::text_iarchive ia(in);
#endif
        if (!in.is_open())
            throw("cannot open " + fname);
        ia >> sd;
        sd.Freeze();
    }
    else{
        throw std::invalid_argument("must have either training corpus or dictionary");
    }

    if (vm.count("hierarchical_ngram_clustering") > 0)
    {
        CorpusWithClassId training = read_corpus_with_classid(vm["train"].as<string>(), sd, kSRC_SOS, kSRC_EOS);

        ptrTrainer->hierarchical_ngram_clustering(vm, training, sd);
    }

    if (vm.count("extract_id_and_string") > 0)
    {
        get_string_and_its_id(vm["train"].as<string>(), make_pair<int, int>(5,4), vm["extract_id_and_string"].as<string>());
    }

    delete ptrTrainer;

    return EXIT_SUCCESS;
}

/**
training on triplet dataset
decoder_t : the type for decoder network, can be RNN or DNN
*/
template <class rnn_t, class TrainProc>
int tuple_main_body(variables_map vm, size_t nreplicate = 0, size_t decoder_additiona_input_to = 0, size_t mem_slots = MEM_SIZE)
{
#ifdef INPUT_UTF8
    kSRC_SOS = sd.Convert(L"<s>");
    kSRC_EOS = sd.Convert(L"</s>");
#else
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
#endif
    verbose = vm.count("verbose");
    g_train_on_turns = vm["turns"].as<int>();

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    TupleCorpus training, devel, testcorpus;
    string line;

    TrainProc  * ptrTrainer = nullptr;

    if (vm.count("readdict"))
    {
        string fname = vm["readdict"].as<string>();
#ifdef INPUT_UTF8
        wifstream in(fname, wifstream::in);
        boost::archive::text_wiarchive ia(in);
#else
        ifstream in(fname, ifstream::in);
        boost::archive::text_iarchive ia(in);
#endif
        ia >> sd;
        sd.Freeze();
    }

    if (vm.count("train") > 0)
    {
        cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
        training = read_tuple_corpus(vm["train"].as<string>(), sd, kSRC_SOS, kSRC_EOS, td, kTGT_SOS, kTGT_EOS, vm["mbsize"].as<int>());
        sd.Freeze(); // no new word types allowed
        td.Freeze();

        training_numturn2did = get_numturn2dialid(training);

        if (vm.count("writesrcdict"))
        {
            string fname = vm["writesrcdict"].as<string>();
            ofstream on(fname);
            boost::archive::text_oarchive oa(on);
            oa << sd;
        }
        if (vm.count("writetgtdict"))
        {
            string fname = vm["writetgtdict"].as<string>();
            ofstream on(fname);
            boost::archive::text_oarchive oa(on);
            oa << td;
        }
    }
    else
    {
        if (vm.count("readtgtdict") == 0 || vm.count("readsrcdict") == 0)
        {
            cerr << "must have either training corpus or dictionary" << endl;
            abort();
        }
        if (vm.count("readsrcdict"))
        {
            string fname = vm["readsrcdict"].as<string>();
            ifstream in(fname);
            boost::archive::text_iarchive ia(in);
            ia >> sd;
        }
        if (vm.count("readtgtdict"))
        {
            string fname = vm["readtgtdict"].as<string>();
            ifstream in(fname);
            boost::archive::text_iarchive ia(in);
            ia >> td;
        }
    }

    LAYERS = vm["layers"].as<int>();
    HIDDEN_DIM = vm["hidden"].as<int>();
    ALIGN_DIM = vm["align"].as<int>();

    string flavour = builder_flavour(vm);
    VOCAB_SIZE_SRC = sd.size();
    VOCAB_SIZE_TGT = td.size();
    nparallel = vm["nparallel"].as<int>();
    mbsize = vm["mbsize"].as < int >();

    if (vm.count("beamsearchdecode"))
    {
        beam_search_decode = vm["beamsearchdecode"].as<int>();
    }

    if (vm.count("devel")) {
        cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
        unsigned min_dev_id = 0;
        devel = read_tuple_corpus(vm["devel"].as<string>(), sd, kSRC_SOS, kSRC_EOS, td, kTGT_SOS, kTGT_EOS, vm["mbsize"].as<int>());
        devel_numturn2did = get_numturn2dialid(devel);
    }

    if (vm.count("testcorpus")) {
        cerr << "Reading test corpus from " << vm["testcorpus"].as<string>() << "...\n";
        unsigned min_dev_id = 0;
        testcorpus = read_tuple_corpus(vm["testcorpus"].as<string>(), sd, kSRC_SOS, kSRC_EOS, td, kTGT_SOS, kTGT_EOS, vm["mbsize"].as<int>());
        test_numturn2did = get_numturn2dialid(testcorpus);
    }

    string fname;
    if (vm.count("parameters")) {
        fname = vm["parameters"].as<string>();
    }
    else {
        ostringstream os;
        os << "attentionwithintention"
            << '_' << LAYERS
            << '_' << HIDDEN_DIM
            << '_' << flavour
            << "-pid" << getpid() << ".params";
        fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    Trainer* sgd = select_trainer<rnn_t, TrainProc>(vm, &model);

    cerr << "%% Using " << flavour << " recurrent units" << endl;

    std::vector<unsigned> dims;
    dims.resize(4);
    if (!vm.count("hidden"))
        dims[ENCODER_LAYER] = HIDDEN_DIM;
    else
        dims[ENCODER_LAYER] = (unsigned)vm["hidden"].as<int>();
    dims[DECODER_LAYER] = dims[ENCODER_LAYER]; /// if not specified, encoder and decoder have the same dimension

    if (!vm.count("align"))
        dims[ALIGN_LAYER] = ALIGN_DIM;
    else
        dims[ALIGN_LAYER] = (unsigned)vm["align"].as<int>();
    if (!vm.count("intentiondim"))
        dims[INTENTION_LAYER] = HIDDEN_DIM;
    else
        dims[INTENTION_LAYER] = (unsigned)vm["intentiondim"].as<int>();


    std::vector<unsigned int> layers;
    layers.resize(4, LAYERS);
    if (!vm.count("intentionlayers"))
        layers[INTENTION_LAYER] = vm["intentionlayers"].as<size_t>();
    rnn_t hred(model, layers, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, mem_slots, vm["scale"].as<cnn::real>());
    prt_model_info<rnn_t, TrainProc>(LAYERS, VOCAB_SIZE_SRC, (const vector<unsigned>&) dims, nreplicate, decoder_additiona_input_to, mem_slots, vm["scale"].as<cnn::real>());

    if (vm.count("initialise"))
    {
        string fname = vm["initialise"].as<string>();
        ifstream in(fname, ifstream::in);
        if (in.is_open())
        {
            boost::archive::text_iarchive ia(in);
            ia >> model;
        }
    }

    ptrTrainer = new TrainProc();

    /*
    if (vm.count("dialogue"))
    {
    if (vm.count("outputfile") == 0)
    {
    cerr << "missing recognition output file" << endl;
    abort();
    }
    ptrTrainer->dialogue(model, hred, vm["outputfile"].as<string>(), sd);
    }
    if (vm.count("nparallel") && !vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
    ptrTrainer->batch_train(model, hred, training, devel, *sgd, fname, vm["epochs"].as<int>(), vm["nparallel"].as<int>());
    }
    */
    if (!vm.count("test") && !vm.count("kbest") && !vm.count("testcorpus"))
    {
        ptrTrainer->train(model, hred, training, *sgd, fname, vm["epochs"].as<int>());
    }
    else if (vm.count("testcorpus"))
    {
        if (vm.count("outputfile") == 0)
        {
            cerr << "missing recognition output file" << endl;
            abort();
        }
        ptrTrainer->test(model, hred, testcorpus, vm["outputfile"].as<string>(), sd, td);
    }

    delete sgd;
    delete ptrTrainer;

    return EXIT_SUCCESS;
}
