#ifndef	_ldaModel_H
#define	_ldaModel_H

/**
    the implementation of LDA refers to the simple LDA in dmlc/experimental-lda
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <cnn/data-util.h>
#include <cnn/dict.h>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

class ldaModel {
public:

	/****** constructor/destructor ******/
	ldaModel(const Corpus& train, const Corpus& test);
	~ldaModel();

	/****** interacting functions ******/
	ldaModel* init(variables_map vm);// initialize the ldaModel randomly
	int train();					// train LDA using prescribed algorithm on training data
	int test(Dict& sd);						// test LDA according to specified method
    int test(Dict& sd, const SentencePair& obs);

    /****** Initialisation aux ******/
    int read_data(const Corpus & training, const Dict& sd, const Corpus& test);				// Read training (and testing) data

    int save_ldaModel_topWords(std::string filename, Dict& sd) const;// ldaModel_name.twords: Top words in each top
    int load_ldaModel(int iter);
    int load_ldaModel(const string & filename);

    int topic_of(int m);

    friend class boost::serialization::access;
    template<class Archive> void save(Archive& ar, const unsigned int version) const {
        ar & alpha;
        ar & beta;
        ar & K; 
        ar & M; 
        ar & V;
        ar & n_iters;

        for (int v = 0; v < V; v++)
            ar & n_wk[v];
        ar & n_k;
    }
    template<class Archive> void load(Archive& ar, const unsigned int version) {
        ar & alpha;
        ar & beta;
        ar & K;
        ar & M;
        ar & V;
        ar & n_iters;

        n_wk.resize(V);
        for (int v = 0; v < V; v++)
            ar & n_wk[v];
        ar & n_k;
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

protected:
	/****** Enums for testing type ldaModel status  ******/
	enum {							// testing types:
		INVALID,					// ldaModel not initialised
		NO_TEST,					// do not report any likelihood
		SELF_TEST,					// report training likelihood
		SEPARATE_TEST				// report likelihood on a held-out testing set
	} testing_type;

	/****** DATA ******/
	Sentences trngdata;				// training dataset
	Sentences testdata;				// test dataset
    Sentences trngresponses;         // responses
    Sentences testresponses;         // responses

    /****** ldaModel Parameters ******/
	int M;							// Number of documents
	int V; 							// Number of words in dictionary
	int K; 							// Number of topics

	/****** ldaModel Hyper-Parameters ******/
	double alpha;					// per document Topic proportions dirichlet prior
	double beta, Vbeta;				// Dirichlet language ldaModel

	/****** ldaModel variables ******/
	int ** z;						// topic assignment for each word
	std::vector<vector<int>> n_wk;					// number of times word w assigned to topic k
	std::vector< std::vector< std::pair<int, int> > > n_mks; //sparse representation of n_mk: number of words assigned to topic k in document m
	std::vector<int> n_k;						// number of words assigned to topic k = sum_w n_wk = sum_m n_mk
		
	/****** Temporary variables ******/
	double * p;
	int *nd_m;
	int *rev_mapper;
      
	/****** Training aux ******/
	int n_iters;	 				// Number of Gibbs sampling iterations
	int n_save;			 			// Number of iters in between saving
	int n_topWords; 				// Number of top words to be printed per topic
	int init_train();					// init for training
	virtual int specific_init() { return 0; }	// if sampling algo need some specific inits
    int sampling(int m);// sampling doc m outsourced to children
    
	/****** Testing aux ******/
	int test_n_iters;
	int test_M;
	int ** test_z;
	int ** test_n_wk;
	int ** test_n_mk;
	int * test_n_k;
	int init_test();				// init for testing
	int vanilla_sampling(int m);	// vanila sampling doc m for testing
    int vanilla_sampling(const Sentence& obs);

	/****** Functions to update sufficient statistics ******/
	inline int add_to_topic(int w, int m, int topic, int old_topic)
	{
        n_wk[w][topic] += 1; 
		if (topic != old_topic && nd_m[topic] == 0)
		{
			rev_mapper[topic] = n_mks[m].size();
			n_mks[m].push_back(std::pair<int, int>(topic, 1));
		}
		else
			n_mks[m][rev_mapper[topic]].second += 1;
		nd_m[topic] += 1;
		if (nd_m[old_topic] == 0)
		{
			n_mks[m][rev_mapper[old_topic]].first = n_mks[m].back().first;
			n_mks[m][rev_mapper[old_topic]].second = n_mks[m].back().second;
			rev_mapper[n_mks[m].back().first] = rev_mapper[old_topic];
			n_mks[m].pop_back();
			rev_mapper[old_topic] = -1;
		}
        n_k[topic] += 1; 

		return 0;
	}
	inline int remove_from_topic(int word, int doc, int topic)
	{
        n_wk[word][topic] -= 1; 
		nd_m[topic] -= 1;
		n_mks[doc][rev_mapper[topic]].second -= 1;
        n_k[topic] -= 1; 

		return 0;
	}

	/****** Performance computations ******/
	std::vector<double> time_ellapsed; // time ellapsed after each iteration
	std::vector<double> likelihood; // likelihood after each iteration
	double newllhw() const;			// per word log-likelihood for new (unseen) data based on the estimated LDA ldaModel
	double llhw() const;			// per word log-likelihood for training data based on the estimated LDA ldaModel

	/****** File and Folder Paths ******/
	std::string ddir;				// data directory
	std::string mdir;				// ldaModel directory
	const Corpus& tfile;				// test data corpus 
    const Corpus& dfile;              // train data corpus

	/****** save LDA ldaModel to files ******/
	int save_ldaModel(int iter) const;						// save ldaModel: call each of the following:		
	int save_ldaModel_time(std::string filename) const;	// ldaModel_name.time: time at which statistics calculated
	int save_ldaModel_llh(std::string filename) const;		// ldaModel_name.llh: Per word likelihood on held out documents
	int save_ldaModel_phi(std::string filename) const;		// ldaModel_name.phi: topic-word distributions

	/****** Some functions for debugging ******/
	int sanity() const;
};

ldaModel::ldaModel(const Corpus& training, const Corpus& test):
tfile(test),
dfile(training)
{
    if (test.size() > 0)
        testing_type = SEPARATE_TEST;
    else
        testing_type = SELF_TEST;

    M = 0;
    V = 0;
    K = 100;

    alpha = 50.0 / K;
    beta = 0.1;

    z = NULL;
    n_wk.clear(); 
    n_k.clear();

    p = NULL;
    nd_m = NULL;
    rev_mapper = NULL;

    n_iters = 1000;
    n_save = 200;
    n_topWords = 0;

    test_n_iters = 10;
    test_M = 0;
    test_z = NULL;
    test_n_wk = NULL;
    test_n_mk = NULL;
    test_n_k = NULL;

    time_ellapsed.reserve(50);
    likelihood.reserve(50);

    ddir = "./";
    mdir = "./";
}

ldaModel::~ldaModel()
{

    if (z)
    {
        for (int m = 0; m < M; m++)
        {
            if (z[m])
            {
                delete[] z[m];
            }
        }
        delete[] z;
    }

    n_wk.clear(); 
    n_k.clear(); 
    
    if (p)		delete[] p;
    if (nd_m)	delete[] nd_m;
    if (rev_mapper)	delete[] rev_mapper;


    if (test_z)
    {
        for (int m = 0; m < test_M; m++)
        {
            if (test_z[m])
            {
                delete[] test_z[m];
            }
        }
        delete[] test_z;
        test_z = nullptr;
    }

    if (test_n_wk)
    {
        for (int w = 0; w < V; w++)
        {
            if (test_n_wk[w])
            {
                delete[] test_n_wk[w];
            }
        }
        delete[] test_n_wk;
        test_n_wk = nullptr;
    }

    if (test_n_mk)
    {
        for (int m = 0; m < test_M; m++)
        {
            if (test_n_mk[m])
            {
                delete[] test_n_mk[m];
            }
        }
        delete[] test_n_mk;
        test_n_mk = nullptr;
    }

    if (test_n_k)	delete[] test_n_k;

}

ldaModel* ldaModel::init(variables_map vm)
{
    double _alpha = -1.0;

    mdir = vm["lda-model"].as<string>();
    alpha = vm["lda-alpha"].as<cnn::real>();
    beta = vm["lda-beta"].as<cnn::real>();
    K = vm["lda-num-topics"].as<int>();
    n_iters = vm["lda-num-iterations"].as<int>();
    n_save = vm["lda-output-state-interval"].as<int>();
    n_topWords = vm["lda-num-top-words"].as<int>();

    //Check specific parameter    
    if (testing_type == SEPARATE_TEST)
    {
        if (tfile.size() == 0)
        {
            throw("Error: test corpus is empty"); 
        }
    }

    std::cout << "data dir = " << ddir << std::endl;
    std::cout << "ldaModel dir = " << mdir << std::endl;
    std::cout << "n_iters = " << n_iters << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "beta = " << beta << std::endl;
    std::cout << "K = " << K << std::endl;

    return this;
}

int ldaModel::train()
{
    std::chrono::high_resolution_clock::time_point ts, tn;
    std::cout << "Sampling " << n_iters << " iterations!" << std::endl;

    init_train();

    for (int iter = 1; iter <= n_iters; ++iter)
    {
        std::cout << "Iteration " << iter << " ..." << std::endl;
        ts = std::chrono::high_resolution_clock::now();

        // for each document
        for (int m = 0; m < M; ++m)
            sampling(m);

        tn = std::chrono::high_resolution_clock::now();
        time_ellapsed.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(tn - ts).count());

#if COMP_LLH
        test(sd);
#endif

        if (n_save > 0)
        {
            if (iter % n_save == 0)
            {
                // saving the ldaModel
                std::cout << "Saving the ldaModel at iteration " << iter << "..." << std::endl;
                save_ldaModel(iter);
            }
        }
    }

    std::cout << "Gibbs sampling completed!" << std::endl;
    std::cout << "Saving the final ldaModel!" << std::endl;
    save_ldaModel(-1);

    return 0;
}

int ldaModel::test(Dict& sd)
{
    init_test();

    if (testing_type == SELF_TEST)
    {
        // just do MAP estimates
        likelihood.push_back(llhw());
        std::cout << "Likelihood on training documents: " << likelihood.back() << " at time " << time_ellapsed.back() << std::endl;
    }
    else if (testing_type == SEPARATE_TEST)
    {
        for (int iter = 1; iter <= test_n_iters; ++iter)
        {
            // for each doc
            for (int m = 0; m < test_M; m++)
                vanilla_sampling(m);
        }
        likelihood.push_back(newllhw());
        std::cout << "Likelihood on held out documents: " << likelihood.back() << std::endl;

        /// compute the topic of each test sentence
        for (int m = 0; m < test_M; m++)
        {
            int this_topic = topic_of(m);
            string ostr = "";
            for (int n = 0; n < testdata[m].size(); n++)
            {
                int w = testdata[m][n];
                string wrd = sd.Convert(w);
                ostr = ostr + wrd + " ";
            }
            cout << ostr << " ||| topic " << this_topic;
            if (testresponses.size() > 0)
            {
                string ostr = "";
                for (auto & w : testresponses[m])
                {
                    string wrd = sd.Convert(w);
                    ostr = ostr + wrd + " ";
                }
                cout << " ||| " << ostr << endl;
            }
        }
    }
    return 0;
}

int ldaModel::test(Dict& sd, const SentencePair& obs)
{
    init_test();

    memset(test_n_wk, 0, sizeof(int)* V * K);
    memset(test_n_mk, 0, sizeof(int)* K);
    memset(test_n_k, 0, sizeof(int)* K);
    for (int n = 0; n < obs.first.size(); n++)
    {
        int w = obs.first[n];
        test_z[0][n] = rand0n_uniform(K - 1);

        int topic = test_z[0][n];

        // number of instances of word i assigned to topic j
        test_n_wk[w][topic] += 1;
        // number of words in document i assigned to topic j
        test_n_mk[0][topic] += 1;
        // total number of words assigned to topic j
        test_n_k[topic] += 1;
    }

    for (int iter = 1; iter <= test_n_iters; ++iter)
    {
        // for each doc
        for (int m = 0; m < test_M; m++)
            vanilla_sampling(obs.first);
    }
    likelihood.push_back(newllhw());
    std::cout << "Likelihood on held out documents: " << likelihood.back() << std::endl;

    /// compute the topic of each test sentence
    for (int m = 0; m < test_M; m++)
    {
        int this_topic = topic_of(m);
        string ostr = "";
        for (int n = 0; n < testdata[m].size(); n++)
        {
            int w = testdata[m][n];
            string wrd = sd.Convert(w);
            ostr = ostr + wrd + " ";
        }
        cout << ostr << " ||| topic " << this_topic;
        if (testresponses.size() > 0)
        {
            string ostr = "";
            for (auto & w : testresponses[m])
            {
                string wrd = sd.Convert(w);
                ostr = ostr + wrd + " ";
            }
            cout << " ||| " << ostr << endl;
        }
    }
    return 0;
}

int ldaModel::read_data(const Corpus & training, const Dict& sd, const Corpus& testing)
{
    flatten_corpus(training, trngdata, trngresponses);
    V = sd.size();
    M = trngdata.size();
    
    flatten_corpus(testing, testdata, testresponses);
    test_M = testdata.size();

    return training.size();
}

int ldaModel::init_train()
{
    Vbeta = V * beta;

    // allocate heap memory for ldaModel variables
    n_wk.resize(V);
    for (int w = 0; w < V; w++)
    {
        n_wk[w] = vector<int>(K, 0); 
    }
    
    n_mks.resize(M);

    n_k.resize(K, 0); 

    // random consistent assignment for ldaModel variables
    z = new int*[M];
    for (int m = 0; m < trngdata.size(); m++)
    {
        int N = trngdata[m].size();
        std::map<int, int > map_nd_m;
        z[m] = new int[N];

        // initialize for z
        for (int n = 0; n < N; n++)
        {
            int topic = rand0n_uniform(K - 1); 
            z[m][n] = topic;
            int w = trngdata[m][n];

            // number of instances of word i assigned to topic j
            n_wk[w][topic] += 1;
            // number of words in document i assigned to topic j
            map_nd_m[topic] += 1;
            // total number of words assigned to topic j
            n_k[topic] += 1; 
        }
        // transfer to sparse representation
        for (auto myc : map_nd_m)
            n_mks[m].push_back(myc);
    }

    time_ellapsed.reserve(n_iters);
    likelihood.reserve(n_iters);

    // allocate heap memory for temporary variables
    p = new double[K];
    nd_m = new int[K];
    rev_mapper = new int[K];
    for (int k = 0; k < K; ++k)
    {
        nd_m[k] = 0;
        rev_mapper[k] = -1;
    }

    return 0;
}

int ldaModel::init_test()
{
    // initialise variables for testing
    Vbeta = V * beta;
    if (test_n_wk == nullptr)
    {
        test_n_wk = new int*[V];
        for (int w = 0; w < V; w++)
        {
            test_n_wk[w] = new int[K];
            for (int k = 0; k < K; k++)
            {
                test_n_wk[w][k] = 0;
            }
        }
    }

    if (test_n_mk == nullptr)
    {
        test_n_mk = new int*[test_M];
        for (int m = 0; m < test_M; m++)
        {
            test_n_mk[m] = new int[K];
            for (int k = 0; k < K; k++)
            {
                test_n_mk[m][k] = 0;
            }
        }
    }

    if (test_n_k == nullptr)
    {
        test_n_k = new int[K];
        for (int k = 0; k < K; k++)
        {
            test_n_k[k] = 0;
        }
    }

    if (test_z == nullptr)
    {
        test_z = new int*[test_M];
        for (int m = 0; m < testdata.size(); m++)
        {
            int N = testdata[m].size();
            test_z[m] = new int[N];

            // assign values for n_wk, n_mk, n_k
            for (int n = 0; n < N; n++)
            {
                int w = testdata[m][n];
                int topic = rand0n_uniform(K - 1);
                test_z[m][n] = topic;

                // number of instances of word i assigned to topic j
                test_n_wk[w][topic] += 1;
                // number of words in document i assigned to topic j
                test_n_mk[m][topic] += 1;
                // total number of words assigned to topic j
                test_n_k[topic] += 1;
            }
        }
    }

    // allocate heap memory for temporary variables
    if (p == nullptr)
        p = new double[K];
    if (nd_m == nullptr)
        nd_m = new int[K];
    if (rev_mapper == nullptr)
        rev_mapper = new int[K];
    for (int k = 0; k < K; ++k)
    {
        nd_m[k] = 0;
        rev_mapper[k] = -1;
    }

    return 0;
}

int ldaModel::sampling(int m)
{
    int kc = 0;
    for (const auto& k : n_mks[m])
    {
        nd_m[k.first] = k.second;
        rev_mapper[k.first] = kc++;
    }
    for (int n = 0; n < trngdata[m].size(); ++n)
    {
        int w = trngdata[m][n];

        // remove z_ij from the count variables
        int topic = z[m][n]; int old_topic = topic;
        remove_from_topic(w, m, topic);

        // do multinomial sampling via cumulative method
        double temp = 0;
        for (int k = 0; k < K; k++)
        {
            temp += (nd_m[k] + alpha) * (n_wk[w][k]  + beta) / (n_k[k] + Vbeta);
            p[k] = temp;
        }

        // scaled sample because of unnormalized p[]
        double u = rand01() * temp;

        // Do a binary search instead!
        topic = std::lower_bound(p, p + K, u) - p;

        // add newly estimated z_i to count variables
        add_to_topic(w, m, topic, old_topic);
        nd_m[topic] += 1;
        z[m][n] = topic;
    }
    for (const auto& k : n_mks[m])
    {
        nd_m[k.first] = 0;
        rev_mapper[k.first] = -1;
    }
    return 0;
}

int ldaModel::vanilla_sampling(int m)
{
    for (int n = 0; n < testdata[m].size(); n++)
    {
        int w = testdata[m][n];

        // remove z_i from the count variables
        int topic = test_z[m][n];
        test_n_wk[w][topic] -= 1;
        test_n_mk[m][topic] -= 1;
        test_n_k[topic] -= 1;

        double psum = 0;
        // do multinomial sampling via cumulative method
        for (int k = 0; k < K; k++)
        {
            psum += (test_n_mk[m][k] + alpha) * (n_wk[w][k] + test_n_wk[w][k] + beta) / (n_k[k]+ test_n_k[k] + Vbeta);
            p[k] = psum;
        }

        // scaled sample because of unnormalized p[]
        double u = rand01() * psum;
        topic = std::lower_bound(p, p + K, u) - p;

        // add newly estimated z_i to count variables
        test_n_wk[w][topic] += 1;
        test_n_mk[m][topic] += 1;
        test_n_k[topic] += 1;
        test_z[m][n] = topic;
    }

    return 0;
}

int ldaModel::vanilla_sampling(const Sentence& obs)
{
    int n = 0;
    for (auto & w : obs)
    {
        // remove z_i from the count variables
        int topic = test_z[0][n];
        test_n_wk[w][topic] -= 1;
        test_n_mk[0][topic] -= 1;
        test_n_k[topic] -= 1;

        double psum = 0;
        // do multinomial sampling via cumulative method
        for (int k = 0; k < K; k++)
        {
            psum += (test_n_mk[0][k] + alpha) * (n_wk[w][k] + test_n_wk[w][k] + beta) / (n_k[k] + test_n_k[k] + Vbeta);
            p[k] = psum;
        }

        // scaled sample because of unnormalized p[]
        double u = rand01() * psum;
        topic = std::lower_bound(p, p + K, u) - p;

        // add newly estimated z_i to count variables
        test_n_wk[w][topic] += 1;
        test_n_mk[0][topic] += 1;
        test_n_k[topic] += 1;
        test_z[0][n] = topic;

        n++;
    }

    return 0;
}

/// get the majority topic of document m
int ldaModel::topic_of(int m)
{
    vanilla_sampling(m);
    std::vector<int> v(K,0);

    for (int n = 0; n < testdata[m].size(); n++)
    {
        int topic = test_z[m][n];
        v[topic] ++;
    }

    std::vector<int>::iterator idx = std::max_element(v.begin(), v.end());
    int topic = std::distance(v.begin(), idx);
    return topic;
}

double ldaModel::newllhw() const
{
    double sum = 0;
    int num_tokens = 0;
    for (int m = 0; m < test_M; ++m)
    {
        double dsum = 0;
        num_tokens += testdata[m].size();
        for (int n = 0; n < testdata[m].size(); n++)
        {
            double wsum = 0;
            int w = testdata[m][n];
            for (int k = 0; k<K; k++)
            {
                wsum += (test_n_mk[m][k] + alpha) * (n_wk[w][k] + test_n_wk[w][k] + beta) / (n_k[k]+ test_n_k[k] + Vbeta);
            }
            dsum += log(wsum);
        }
        sum += dsum - testdata[m].size()*log(testdata[m].size() + K * alpha);
    }
    return sum / num_tokens;
}

double ldaModel::llhw() const
{
    double sum = 0;
    int num_tokens = 0;
    for (int m = 0; m < M; m++)
    {
        for (auto k = n_mks[m].begin(); k != n_mks[m].end(); ++k)
        {
            nd_m[k->first] = k->second;
        }

        double dsum = 0;
        num_tokens += trngdata[m].size();
        for (int n = 0; n < trngdata[m].size(); n++)
        {
            double wsum = 0;
            int w = trngdata[m][n];
            for (int k = 0; k<K; k++)
            {
                wsum += (nd_m[k] + alpha) * (n_wk[w][k] + beta) / (n_k[k]+ Vbeta);
            }
            wsum /= (trngdata[m].size() + K * alpha);
            dsum += log(wsum);
        }
        sum += dsum;
        for (auto k = n_mks[m].begin(); k != n_mks[m].end(); ++k)
        {
            nd_m[k->first] = 0;
        }
    }
    return sum / num_tokens;
}

int ldaModel::save_ldaModel(int iter) const
{
    std::string ldaModel_name = mdir + "-";
    if (iter >= 0)
    {
        std::ostringstream sstr1;
        sstr1 << std::setw(5) << std::setfill('0') << iter;
        ldaModel_name += sstr1.str();
    }
    else
    {
        ldaModel_name += "final";
    }

    if (save_ldaModel_time(ldaModel_name + ".time"))
    {
        return 1;
    }
    std::cout << "time done" << std::endl;
    if (save_ldaModel_llh(ldaModel_name + ".llh"))
    {
        return 1;
    }
    std::cout << "llh done" << std::endl;

    string fname = ldaModel_name + ".mdl";
    ofstream on(fname);
    boost::archive::text_oarchive oa(on);
    oa << *this;

    std::cout << "others done" << std::endl;
    if (n_topWords > 0)
    {
        //if (save_ldaModel_twords(mdir + ldaModel_name + ".twords")) 
        //ff{
        //	return 1;
        //}
        //std::cout << "twords done" << std::endl;
    }
    //if (ldaModel_status == ldaModel_SELF_TEST)
    //{
    //	if (save_ldaModel_phi(mdir + ldaModel_name + ".phi"))
    //	{
    //		return 1;
    //	}
    //	std::cout << "phi done" << std::endl;
    //}
    return 0;
}

int ldaModel::load_ldaModel(int iter)
{
    std::string ldaModel_name = mdir + "-";
    if (iter >= 0)
    {
        std::ostringstream sstr1;
        sstr1 << std::setw(5) << std::setfill('0') << iter;
        ldaModel_name += sstr1.str();
    }
    else
    {
        ldaModel_name += "final";
    }

    string fname = ldaModel_name + ".mdl";
    ifstream in(fname);
    boost::archive::text_iarchive ia(in);
    ia >> *this;

    return 0;
}

int ldaModel::load_ldaModel(const string & filename)
{
    string fname = filename;
    ifstream in(fname);
    if (!in.is_open())
    {
        cerr << "cannot open " << fname << endl;
        throw("cannot open " + fname);
    }
    boost::archive::text_iarchive ia(in);
    ia >> *this;

    return 0;
}

int ldaModel::save_ldaModel_time(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
    {
        std::cout << "Error: Cannot open file to save: " << filename << std::endl;
        return 1;
    }

    for (unsigned r = 0; r < time_ellapsed.size(); ++r)
    {
        fout << time_ellapsed[r] << std::endl;
    }

    fout.close();

    return 0;
}

int ldaModel::save_ldaModel_llh(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
    {
        std::cout << "Error: Cannot open file to save: " << filename << std::endl;
        return 1;
    }

    for (unsigned r = 0; r < likelihood.size(); ++r)
    {
        fout << likelihood[r] << std::endl;
    }

    fout.close();

    return 0;
}

int ldaModel::save_ldaModel_topWords(std::string filename, Dict& sd) const
{
    std::ofstream fout(filename);
    if (!fout)
    {
        std::cout << "Error: Cannot open file to save: " << filename << std::endl;
        return 1;
    }

    int _n_topWords = n_topWords;
    if (_n_topWords > V)	_n_topWords = V;

    std::map<int, std::string>::const_iterator it;

    for (int k = 0; k < K; k++)
    {
        std::vector<std::pair<int, int> > words_probs(V);
        std::pair<int, int> word_prob;
        for (int w = 0; w < V; w++)
        {
            word_prob.first = w;
            word_prob.second = n_wk[w][k];
            words_probs[w] = word_prob;
        }

        // quick sort to sort word-topic probability
        std::sort(words_probs.begin(), words_probs.end());

        fout << "Topic " << k << "th:" << std::endl;
        for (int i = 0; i < _n_topWords; i++)
        {
            string it = sd.Convert(words_probs[i].first);
            fout << "\t" << it << "   " << words_probs[i].second << std::endl;
        }
    }

    fout.close();

    return 0;
}

int ldaModel::save_ldaModel_phi(std::string filename) const
{
    std::ofstream fout(filename);
    if (!fout)
    {
        std::cout << "Error: Cannot open file to save: " << filename << std::endl;
        return 1;
    }

    for (int k = 0; k < K; k++)
    {
        for (int w = 0; w < V; w++)
        {
            fout << (n_wk[w][k] + beta) / (n_k[k]+ Vbeta) << " ";
        }
        fout << std::endl;
    }

    fout.close();

    return 0;
}

int ldaModel::sanity() const
{
    long tott = 0;
    for (int m = 0; m < M; ++m)
    {
        int sumd = 0;
        for (const auto &t : n_mks[m])
            sumd += t.second;
        if (sumd == trngdata[m].size())
            tott += sumd;
        else
            std::cout << "Length mismatch at doc: " << m << std::endl;
    }
    std::cout << "Total number of training tokens: " << tott << std::endl;
    return 0;
}


#endif