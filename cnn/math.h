#pragma once

#include <initializer_list>
#include <vector>

#include "cnn/cnn.h"
#include "cnn/random.h"

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "cnn/cuda.h"
#endif
#include <boost/serialization/array.hpp>

namespace cnn {

    cnn::real rand01();
    int rand0n(int n);
    cnn::real rand_normal();
    int rand0n_uniform(int n);
    std::vector<int> rand0n_uniform(int vecsize, int n_exlusive);

    /// sample_dist : the sample distribution, size = vocab_size
    std::vector<int> rand0n_uniform(int vecsize, int n_exclusive, const std::vector<cnn::real>& sample_dist);

} // namespace cnn

