#include "cnn/math.h"

#include <random>
#include <vector>
#include <cstring>
#include <boost/random/discrete_distribution.hpp>

#if HAVE_CUDA
#include "cnn/cuda.h"
#endif

using namespace std;

namespace cnn {

    extern mt19937* rndeng;

    cnn::real rand01() {
      uniform_real_distribution<cnn::real> distribution(0, 1);
      return distribution(*rndeng);
    }

    int rand0n(int n) {
      assert(n > 0);
      int x = rand01() * n;
      while(n == x) { x = rand01() * n; }
      return x;
    }

    cnn::real rand_normal() {
      normal_distribution<cnn::real> distribution(0, 1);
      return distribution(*rndeng);
    }

    int rand0n_uniform(int n)
    {
        assert(n > 0);
        uniform_int_distribution<> distribution(0, n);
        return distribution(*rndeng);
    }

    std::vector<int> rand0n_uniform(int vecsize, int n_exclusive)
    {
        std::vector<int> res(vecsize);
        for (int i = 0; i < vecsize; i++)
            res[i] = rand0n_uniform(n_exclusive);

        return res;
    }

    std::vector<int> rand0n_uniform(int vecsize, int n_exclusive, const std::vector<cnn::real>& sample_dist)
    {
        std::vector<int> res(vecsize);
        boost::random::discrete_distribution<int> d(sample_dist.begin(), sample_dist.end());
        for (int i = 0; i < vecsize; i++)
            res[i] = d(*rndeng);

        return res;
    }

} // namespace cnn
