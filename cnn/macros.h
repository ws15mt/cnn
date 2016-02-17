#pragma once

namespace cnn {

//    #define USE_DOUBLE
#ifdef USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

#define ENCODER_LAYER 0
#define INTENTION_LAYER 1
#define DECODER_LAYER 2  
#define ALIGN_LAYER 3

#define INPUT_LAYER ENCODER_LAYER
#define HIDDEN_LAYER INTENTION_LAYER
#define OUTPUT_LAYER DECODER_LAYER


/// this is for ngram models
#define MIN_OCC_COUNT 20

/// assume column major
#define IDX2C(i, j, ld) (((j) * (ld)) + (i)) // 0 based indexing

/// for GPU
#define MAX_THREADS_PER_BLOCK 512

///for gradient checking
#define GRADIENT_CHECK_DIGIT_SIGNIFICANT_LEVEL 5
#define GRADIENT_CHECK_PARAM_DELTA 1e-5

};
