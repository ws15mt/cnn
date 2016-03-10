#include "cnn/init.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/cnn.h"
#include "cnn/model.h"
#include <iostream>
#include <random>
#include <cmath>

#if HAVE_CUDA
#include "cnn/cuda.h"
#include <device_launch_parameters.h>
#endif

using namespace std;

namespace cnn {
    
    AlignedMemoryPool<ALIGN>* fxs = nullptr;
    AlignedMemoryPool<ALIGN>* dEdfs = nullptr;
    AlignedMemoryPool<ALIGN>* mem_nodes= nullptr;   /// for nodes allocation/delocation. operation of new/delete of each node has been overwritten to use this memory pool for speed-up
    mt19937* rndeng = nullptr;

    char* getCmdOption(char ** begin, char ** end, const std::string & option)
    {
        char ** itr = std::find(begin, end, option);
        if (itr != end && ++itr != end)
        {
            return *itr;
        }
        return 0;
    }

	static void RemoveArgs(int& argc, char**& argv, int& argi, int n) {
	  for (int i = argi + n; i < argc; ++i)
	    argv[i - n] = argv[i];
	  argc -= n;
	  assert(argc >= 0);
	}
	
    bool cmdOptionExists(char** begin, char** end, const std::string& option)
    {
        return std::find(begin, end, option) != end;
    }

    void Initialize(int& argc, char**& argv, unsigned random_seed, bool demo) {
        cerr << "Initializing...\n";
#if HAVE_CUDA
        Initialize_GPU(argc, argv);
#else
        kSCALAR_MINUSONE = (cnn::real*)cnn_mm_malloc(sizeof(cnn::real), CNN_ALIGN);
        *kSCALAR_MINUSONE = -1;
        kSCALAR_ONE = (cnn::real*)cnn_mm_malloc(sizeof(cnn::real), CNN_ALIGN);
        *kSCALAR_ONE = 1;
        kSCALAR_ZERO = (cnn::real*)cnn_mm_malloc(sizeof(cnn::real), CNN_ALIGN);
        *kSCALAR_ZERO = 0;
#endif

        if (random_seed == 0)
        {
            if (cmdOptionExists(argv, argv + argc, "--seed"))
            {
                string seed = getCmdOption(argv, argv + argc, "--seed");
                stringstream(seed) >> random_seed;
            }
            else
            {
                random_device rd;
                random_seed = rd();
            }
        }
        rndeng = new mt19937(random_seed);

        cerr << "Allocating memory...\n";
		unsigned long num_mb = 512UL;
        mem_nodes = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20), true);
        if (demo)
        {
            fxs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
            dEdfs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
        }
        else
        {
#ifdef HAVE_CUDA
            fxs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
            dEdfs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 20));
#else
            fxs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 22));
            dEdfs = new AlignedMemoryPool<ALIGN>(512UL * (1UL << 22));
#endif
        }
        cerr << "Done.\n";
    }

  void Free() 
  {
        cerr << "Freeing memory ...\n";
        cnn_mm_free(kSCALAR_MINUSONE);
        cnn_mm_free(kSCALAR_ONE);
        cnn_mm_free(kSCALAR_ZERO);

        delete (rndeng); 
        delete (fxs);
        delete (dEdfs);
        delete (mem_nodes);

        for (auto p : kSCALAR_ONE_OVER_INT)
            cnn_mm_free(p);

#ifdef HAVE_CUDA
        Free_GPU();
#endif
        cerr << "Done.\n";
  }

} // namespace cnn
