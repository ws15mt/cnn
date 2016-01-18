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
#include "cnn/cnn-helper.h"
#include "cnn/expr-xtra.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;

unsigned int LAYERS = 2;
unsigned int INPUT_DIM = 8;  //256
unsigned int HIDDEN_DIM = 24;  // 1024
unsigned int VOCAB_SIZE = 0;

int verbose = 0; 

cnn::Dict d;
int kSOS;
int kEOS;

template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_c;
  vector<Parameters*> p_R;
  vector<Parameters*> p_bias;
  Parameters* p_cls, *p_cls_bias;
  unsigned int ncls;
  Builder builder;
  vector<int> clssize;
  vector<long> word2cls;
  vector<long> acc_cls2size;
  vector<long> dict_wrd_id2within_class_id;
  explicit RNNLanguageModel(const vector<int>& cls2nbrwords, /// #words for each class, class starts from 0
      const vector<long> & acc_cls2size, /// the accumulated class size
      const vector<long>& word2cls, 
      const vector<long>& dict_wrd_id2within_class_id,
      Model& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
      clssize(cls2nbrwords), word2cls(word2cls), acc_cls2size(acc_cls2size), dict_wrd_id2within_class_id(dict_wrd_id2within_class_id)
  {
      unsigned int n_cls = clssize.size();
      p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
      p_cls = model.add_parameters({ n_cls, HIDDEN_DIM });
      p_cls_bias = model.add_parameters({ n_cls});
      for (size_t id = 0; id < n_cls; id++)
      {
          unsigned int  clssize = cls2nbrwords[id];
          p_R.push_back(model.add_parameters({ clssize, HIDDEN_DIM }));
          p_bias.push_back(model.add_parameters({ clssize }));
      }
  }

  // return Expression of total loss
  Expression BuildLMGraph(const vector<int>& sent, ComputationGraph& cg) {
    const unsigned slen = sent.size() - 1;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    
    Expression i_cls, i_cls_bias;
    i_cls = parameter(cg, p_cls); 
    i_cls_bias = parameter(cg, p_cls_bias);

    vector<Expression> i_R, i_bias;
    for (auto& p : p_R)
        i_R.push_back(parameter(cg, p)); // hidden -> word rep parameter
    for (auto& p: p_bias)
        i_bias.push_back(parameter(cg, p));  // word bias

    vector<Expression> errs;
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      int cls_id = word2cls[sent[t + 1]];
      Expression i_r_t =  i_bias[cls_id] + i_R[cls_id] * i_y_t;
      Expression i_c_t = i_cls_bias + i_cls * i_y_t;
      if (verbose)
		  display_value(i_r_t, cg, "response at " + t);
    
#if 1
      Expression i_err_cls = pick(log_softmax(i_c_t), cls_id);
      Expression i_err_prb = pick(log_softmax(i_r_t), dict_wrd_id2within_class_id[sent[t + 1]]);
      errs.push_back(i_err_cls + i_err_prb);
#if 0
      Expression i_ydist = softmax(i_r_t);
      i_ydist = log(i_ydist)
      errs.push_back(pick(i_ydist, sent[t+1]));
#endif
#else
      ppl turns to be 0 at epoch 4, so there is a bug
      Expression i_err_cls = pickneglogsoftmax(i_c_t, cls_id);
      Expression i_err_prb = pickneglogsoftmax(i_r_t, dict_wrd_id2within_class_id[sent[t + 1]]);
      errs.push_back(i_err_cls + i_err_prb);
#endif
    }
    Expression i_nerr = sum(errs);
#if 1
    return -i_nerr;
#else
    return i_nerr;
#endif
  }

};

template <class LM_t>
void train(Model &model, LM_t &lm,
    const vector<vector<int>>& training,
    const vector<vector<int>>& dev,
    Trainer *sgd, const string& fname,
    bool randomSample)
{
    cnn::real best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500;
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    unsigned total_epoch = 40;

    ofstream out(fname, ofstream::out);
    boost::archive::text_oarchive oa(out);
    oa << model;
    out.close();

    size_t i_epoch = 0;
    while (sgd->epoch < total_epoch) {
        Timer iteration("completed in");
        cnn::real loss = 0;
        unsigned chars = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) {
                si = 0;
                if (first) { first = false; }
                else { sgd->update_epoch(); }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance
            ComputationGraph cg;
            auto& sent = training[order[si]];
            chars += sent.size() - 1;
            ++si;
            lm.BuildLMGraph(sent, cg);
            loss += as_scalar(cg.forward());
            cg.backward();
            sgd->update();
            ++lines;
        }
        sgd->status();
        cerr << " report = " << report << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
        iteration.WordsPerSecond(chars);
        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            cnn::real dloss = 0;
            int dchars = 0;
            for (auto& sent : dev) {
                ComputationGraph cg;
                lm.BuildLMGraph(sent, cg);
                dloss += as_scalar(cg.forward());
                dchars += sent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(fname, ofstream::out);
                boost::archive::text_oarchive oa(out);
                oa << model;
                out.close();
            }
            else{
                sgd->eta *= 0.5;
            }
            cerr << "\n***TEST E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }

        i_epoch++;
    }
}

template <class LM_t>
void testcorpus(Model &model, LM_t &lm,
    const vector<vector<int>>& dev)
{
    unsigned lines = 0;
    cnn::real dloss = 0;
    int dchars = 0;
    for (auto& sent : dev) {
        ComputationGraph cg;
        lm.BuildLMGraph(sent, cg);
        dloss += as_scalar(cg.forward());
        dchars += sent.size() - 1;
    }

    cerr << "\n***DEV [epoch=" << (lines / (cnn::real)dev.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
}

void initialise(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    ifstream in(filename, ifstream::in);
    boost::archive::text_iarchive ia(in);
    ia >> model;
}

/**
dict_wrd_id2within_class_id : the dictionary word id to the id inside a class
*/
void load_word2cls_fn(string word2clsfn, Dict& sd, std::vector<long>& wrd2cls, std::vector<long>& dict_wrd_id2within_class_id)
{
    ifstream in(word2clsfn);
    string line;

    wrd2cls.resize(sd.size());
    dict_wrd_id2within_class_id.resize(sd.size());
    map<int, int> cls2acccnt; /// the count for each class so far
    while (getline(in, line)) {

        std::istringstream in(line);
        std::string word;
        string cls;

        in >> word;
        in >> cls;

        int icls = boost::lexical_cast<int>(cls) - 1;
        int wridx = sd.Convert(word);

        wrd2cls[wridx] = icls;
        if (cls2acccnt.find(icls) == cls2acccnt.end())
            cls2acccnt[icls] = 1;
        else
            cls2acccnt[icls] += 1;
        dict_wrd_id2within_class_id[wridx] = cls2acccnt[icls] - 1;
    }
    in.close();
}

/**
acc_cls2size : the accumulated class size
*/
void load_clssize_fn(string clsszefn, std::vector<int> & cls2size, std::vector<long>& acc_cls2size)
{
    ifstream in(clsszefn);
    string line;

    cls2size.clear(); 
    acc_cls2size.clear();
    int idx = 1;

    while (getline(in, line)) {

        std::istringstream in(line);
        std::string cls;
        string sze;

        in >> cls;
        in >> sze;

        int icls = boost::lexical_cast<int>(cls) - 1;
        if (icls != idx - 1)
            throw("class id should start from 1 and then consecuitively increasing with step 1");

        cls2size.push_back(boost::lexical_cast<int>(sze));
        if (acc_cls2size.size() == 0)
            acc_cls2size.push_back(cls2size.back());
        else
            acc_cls2size.push_back(acc_cls2size.back() + cls2size.back());

        idx++;
    }
    in.close();
}

bool check_info_correct(Dict& sd, const std::vector<long>& wrd2cls, const std::vector<long>& dict_wrd_id2within_class_id, const std::vector<int> & cls2size, const std::vector<long>& acc_cls2size)
{
    for (auto&p : sd.GetWordList())
    {
        long wd = sd.Convert(p);
        int  cls = wrd2cls[wd];
        int clssize = cls2size[cls];
        int pos = dict_wrd_id2within_class_id[wd];
        if (pos < 0 || pos >= clssize)
        {
            cerr << "word " << p << " id " << wd << " cls " << cls << " clssize " << clssize << " pos in cls " << pos << " wrong" << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // command line processing
  using namespace boost::program_options;
  variables_map vm;
  options_description opts("Allowed options");
  opts.add_options()
      ("help", "print help message")
      ("seed,s", value<int>()->default_value(217), "random seed number")
      ("train,t", value<string>(), "file containing training sentences")
      ("devel,d", value<string>(), "file containing development sentences.")
      ("test,T", value<string>(), "file containing testing source sentences")
      ("word2cls", value<string>(), "word2class info file")
      ("cls2size", value<string>(), "class size information file")
      ("initialise,i", value<string>(), "load initial parameters from file")
      ("parameters,p", value<string>(), "save best parameters to this file")
      ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
      ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
      ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
      ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
      ("dglstm", "use depth-gated LSTM for recurrent structure; default RNN")
      ("verbose,v", "be extremely chatty")
      ("generate,g", value<bool>()->default_value(false), "generate random samples")
      ;
  store(parse_command_line(argc, argv, opts), vm);

  string flavour;
  if (vm.count("gru"))	flavour = "gru";
  else if (vm.count("lstm"))	flavour = "lstm";
  else if (vm.count("rnnem"))	flavour = "rnnem";
  else if (vm.count("dglstm")) flavour = "dglstm";
  else if (vm.count("nmn")) flavour = "nmn";
  else			flavour = "rnn";


  LAYERS = vm["layers"].as<int>();
  HIDDEN_DIM = vm["hidden"].as<int>();

  bool generateSample = false;
  generateSample = vm["generate"].as<bool>();

  string fname;
  if (vm.count("parameters")) {
      fname = vm["parameters"].as<string>();
  }
  else {
      ostringstream os;
      os << "lm"
          << '_' << LAYERS
          << '_' << HIDDEN_DIM
          << '_' << flavour
          << "-pid" << getpid() << ".params";
      fname = os.str();
  }

  cerr << "Parameters will be written to: " << fname << endl;

  if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && vm.count("test") != 1)) {
      cout << opts << "\n";
      return 1;
  }

  kSOS = d.Convert("<s>");
  kEOS = d.Convert("</s>");
  vector<vector<int>> training, dev, test;
  string line;
  int tlc = 0;
  int ttoks = 0;

  string infile = vm["train"].as<string>();
  cerr << "Reading training data from " << infile << "...\n";

  {
    ifstream in(infile);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      training.push_back(ReadSentence(line, &d));
      ttoks += training.back().size();
      if (training.back().front() != kSOS && training.back().back() != kEOS) {
		  throw("Training sentence in %s : %d didnt start or end with <s>, </s>", infile.c_str(), tlc );
      }
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.Freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();

  if (vm.count("devel") > 0)
  {
      int dlc = 0;
      int dtoks = 0;
      string devfile = vm["devel"].as<string>();
      cerr << "Reading training data from " << devfile << "...\n";
      {
          ifstream in(devfile);
          assert(in);
          while (getline(in, line)) {
              ++dlc;
              dev.push_back(ReadSentence(line, &d));
              dtoks += dev.back().size();
			  if (dev.back().front() != kSOS && dev.back().back() != kEOS) {
				  throw("Dev sentence in %s : %d didn't start or end with <s>, </s> ", devfile.c_str(), tlc);
			  }
          }
          cerr << dlc << " lines, " << dtoks << " tokens\n";
      }
  }

  Model model;
  bool use_momentum = false;
  Trainer* sgd = nullptr;
  if (use_momentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);

  std::vector<long> wrd2cls;
  std::vector<int> cls2size;
  std::vector<long> acc_cls2size;
  std::vector<long> dict_wrd_id2within_class_id;
  if (vm["word2cls"].as<string>() != "")
  {
      load_word2cls_fn(vm["word2cls"].as<string>(), d, wrd2cls, dict_wrd_id2within_class_id);
      load_clssize_fn(vm["cls2size"].as<string>(), cls2size, acc_cls2size);
      check_info_correct(d, wrd2cls, dict_wrd_id2within_class_id, cls2size, acc_cls2size);
  }
  else{
      throw std::invalid_argument("need to specify word2cls and cls2size files for word clustering information.");
  }

  if (vm.count("test") == 0)
  {
      if (vm.count("lstm")) {
          cerr << "%% Using LSTM recurrent units" << endl;
          RNNLanguageModel<LSTMBuilder> lm(cls2size, acc_cls2size, wrd2cls, dict_wrd_id2within_class_id, model);
          train(model, lm, training, dev, sgd, fname, generateSample);
      }
      else if (vm.count("dglstm")) {
          cerr << "%% Using DGLSTM recurrent units" << endl;
          RNNLanguageModel<DGLSTMBuilder> lm(cls2size, acc_cls2size, wrd2cls, dict_wrd_id2within_class_id, model);
          train(model, lm, training, dev, sgd, fname, generateSample);
      }
  }
  else
  {
      string testfile = vm["test"].as<string>();
      int dlc = 0;
      int dtoks = 0;
      cerr << "Reading training data from " << testfile << "...\n";
      {
          ifstream in(testfile);
          assert(in);
          while (getline(in, line)) {
              ++dlc;
              test.push_back(ReadSentence(line, &d));
              dtoks += test.back().size();
			  if (test.back().front() != kSOS && test.back().back() != kEOS) {
				  throw("Dev sentence in %s : %d didnt start or end with <s>, </s> ", testfile.c_str(), tlc);
			  }
          }
          cerr << dlc << " lines, " << dtoks << " tokens\n";
      }

      if (vm.count("test"))
      {
          if (vm.count("lstm")){
              cerr << "%% using LSTM recurrent units" << endl;
              RNNLanguageModel<LSTMBuilder> lm(cls2size, acc_cls2size, wrd2cls, dict_wrd_id2within_class_id, model);
              if (vm.count("initialise"))
                  initialise(model, vm["initialise"].as<string>());
              testcorpus(model, lm, test);
          }
          if (vm.count("dglstm")){
              cerr << "%% using DGLSTM recurrent units" << endl;
              RNNLanguageModel<DGLSTMBuilder> lm(cls2size, acc_cls2size, wrd2cls, dict_wrd_id2within_class_id, model);
              if (vm.count("initialise"))
                  initialise(model, vm["initialise"].as<string>());
              testcorpus(model, lm, test);
          }
      }
  }

  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (argc == 4) {
    string fname = argv[3];
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }

  delete sgd;
}

