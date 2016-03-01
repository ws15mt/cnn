#ifndef CNN_GPU_OPS_H
#define CNN_GPU_OPS_H

#include <cnn/macros.h>

namespace cnn {
namespace gpu {

    void vpairwise_rank_loss(int n, cnn::real margin, const cnn::real* xgood, const cnn::real* xbad, cnn::real* y);

    void set_to_value_of(int n, cnn::real* x0, cnn::real val);
    void set_to_value_of(int n, cnn::real* x0, cnn::real *val);

/// for convlution networks
    void conv1dwide(const int n, const int m, const cnn::real* xs, const int k, const cnn::real *fx, cnn::real *fy);
    void conv1dwide_backward(const int i, const int n, const int m, const cnn::real* xs, const int k, const cnn::real *fx, const cnn::real* dEdf, cnn::real *dEdx);

    /// add bias
    void addVectorToAllColumns(const int n, const cnn::real * xs, const int m, const cnn::real* fx, cnn::real *fy);
    void addVectorToAllColumns_backward(const int i, const int r, const int c, const cnn::real* dEdf, cnn::real *dEdxi);

    void foldRows(const int n, const int m, const cnn::real *xs, const int stride, const int orows, cnn::real *fy);
    void foldRows_backward(const int orows, const cnn::real* dEdf, const int n, const int m, cnn::real *fy);

    void kMaxPooling(const int n, const int m, const cnn::real *xs, const int k, cnn::real *fy, int* aux_mem);
    void kMaxPooling_backward(const int n, const int m, const cnn::real *xs, const int k, const cnn::real * dEdf, cnn::real *dEdxi, const int* aux_mem);

    void vpairwise_rank_loss(int n, cnn::real margin, const cnn::real* xgood, const cnn::real* xbad, cnn::real* y);
    void vpairwise_rank_loss_backward(int n, bool d_wrt_correct, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vcwise_product(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y);
    void vcwise_product_backward(int n, const cnn::real* dEdy, const cnn::real* x_other, cnn::real* dEdx);
    void vcwise_quotient(int n, const cnn::real* x0, const cnn::real* x1, cnn::real* y);
    void vcwise_quotient_backward(int n, const cnn::real* dEdy, const cnn::real* x_other, cnn::real* dEdx);
    void vconstant_minusx(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    /// c should be zero if used as back-propagation of y = x - c, since dx += dy should be the gradient to x
    void vconstant_minusx_backward(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    void vconstant_multiplyx(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    void vconstant_multiplyx_backward(int n, cnn::real c, const cnn::real* x, cnn::real* y);
    void vnegate(int n, const cnn::real* x, cnn::real* y);
    void vnegate_backward(int n, const cnn::real* dEdf, cnn::real* dEdx);
    void vrelu(int n, const cnn::real* x, cnn::real* y);
    void vrelu_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vexponential_linear_units(int n, const cnn::real* x, const cnn::real scale, cnn::real* y);
    void vexponential_linear_units_backward(int n, const cnn::real* fx, const cnn::real* dEdf, const cnn::real scale, cnn::real* dEdx);
    void vexp(int n, const cnn::real* x, cnn::real* y);
    void vtanh(int n, const cnn::real* x, cnn::real* y);
    void vtanh_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vlog(int n, const cnn::real* x, cnn::real* y);
    void vlog_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void vlogistic(int n, const cnn::real* x, cnn::real* y);
    void vlogistic_backward(int n, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx);
    void l2_norm_reducer(int n, const cnn::real* x0, cnn::real* y, bool square, bool accumulate);
    void sqeucdist(int n, const cnn::real* x0, const cnn::real *x1, cnn::real* y);
    void sqeucdist_backward(int n, const cnn::real* dEdy, const cnn::real* x0, const cnn::real* x1, cnn::real* dEdx, int i);
    void pnlsoftmax(int n, int elem_idx, const cnn::real* x0, cnn::real* y, cnn::real* logz);
    void pnlsoftmax_backward(int n, int elem_idx, const cnn::real* x0, const cnn::real* dEdf, const cnn::real* logz, cnn::real* dEdx);
    void logsoftmax(int row, int col, const cnn::real* x0, cnn::real* y);
    void logsoftmax_backward(int row, int col, const cnn::real* fx, const cnn::real* dEdf, cnn::real* dEdx, cnn::real *softmax, cnn::real* row_sum_grd);
    void softmax(int row, int col, const cnn::real* x0, cnn::real* y);
    void softmax_backward(int row, int col, const cnn::real *fx, const cnn::real *dEdf, cnn::real *dEdx);
    void sgd_update(int n, const cnn::real* g, cnn::real* x, cnn::real scale, cnn::real lambda);
    void sgd_momentum_update(int n, const cnn::real* g, cnn::real* x, cnn::real * v, cnn::real scale, cnn::real lambda, cnn::real momentum);
    void rmsprop_momentum_update(int n, const cnn::real* g, cnn::real* x, cnn::real* v, cnn::real *r, cnn::real scale, cnn::real lambda, cnn::real momentum, cnn::real rho, cnn::real epsilon);

    void vector_sum(int rows, int cols, const cnn::real * a, cnn::real* c, const bool isColWise);
    void vector_add_const(int rows, int cols, const cnn::real * a, int brow, int bcol, const cnn::real* b, cnn::real * c, bool isColWise);


} // namespace gpu
} // namespace cnn

#endif
