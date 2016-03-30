#include "cnn/expr.h"

#include <initializer_list>

#include "cnn/nodes.h"
#include "cnn/conv.h"

namespace cnn { namespace expr {

Expression input(ComputationGraph& g, cnn::real s) { return Expression(&g, g.add_input(s)); }
Expression input(ComputationGraph& g, const cnn::real *ps) { return Expression(&g, g.add_input(ps)); }
Expression input(ComputationGraph& g, const Dim& d, const std::vector<cnn::real>& pdata) { return Expression(&g, g.add_input(d, pdata)); }
Expression input(ComputationGraph& g, const Dim& d, const std::vector<cnn::real>* pdata) { return Expression(&g, g.add_input(d, pdata)); }
Expression reference(ComputationGraph& g, const Dim& d, const cnn::real* pdata) { return Expression(&g, g.add_reference(d, pdata)); }
Expression const_parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_const_parameters(p)); }
Expression parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_parameters(p)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, unsigned index) { return Expression(&g, g.add_lookup(p, index)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const std::vector<unsigned>& indices) { return Expression(&g, g.add_lookup(p, indices)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const std::vector<unsigned>* pindices) { return Expression(&g, g.add_lookup(p, pindices)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, unsigned index) { return Expression(&g, g.add_const_lookup(p, index)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex) { return Expression(&g, g.add_const_lookup(p, pindex)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, const std::vector<unsigned>& indices) { return Expression(&g, g.add_const_lookup(p, indices)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, const std::vector<unsigned>* pindices) { return Expression(&g, g.add_const_lookup(p, pindices)); }
Expression zeroes(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<Zeroes>(d)); }

Expression operator-(const Expression& x) { return Expression(x.pg, x.pg->add_function<Negate>({x.i})); }
Expression operator+(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Sum>({x.i, y.i})); }
Expression operator+(cnn::real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantPlusX>({y.i}, x)); }
Expression operator+(const Expression& x, cnn::real y) { return y+x; }
Expression operator-(const Expression& x, const Expression& y) { return x+(-y); }
Expression operator-(cnn::real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantMinusX>({y.i}, x)); }
Expression operator-(const Expression& x, cnn::real y) { return -(y-x); }
Expression operator*(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<MatrixMultiply>({x.i, y.i})); }
Expression operator*(const Expression& x, cnn::real y) { return Expression(x.pg, x.pg->add_function<ConstScalarMultiply>({x.i}, y)); }
Expression cdiv(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<CwiseQuotient>({x.i, y.i})); }
Expression colwise_add(const Expression& x, const Expression& bias) { return Expression(x.pg, x.pg->add_function<AddVectorToAllColumns>({x.i, bias.i})); }
Expression contract3d_1d(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i})); }
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>({x.i, y.i, b.i})); }

Expression sqrt(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sqrt>({x.i})); }
//Expression erf(const Expression& x) { return Expression(x.pg, x.pg->add_function<Erf>({x.i})); }
Expression tanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tanh>({x.i})); }
//Expression lgamma(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogGamma>({x.i})); }
Expression log(const Expression& x) { return Expression(x.pg, x.pg->add_function<Log>({x.i})); }
Expression exp(const Expression& x) { return Expression(x.pg, x.pg->add_function<Exp>({x.i})); }
Expression square(const Expression& x) { return Expression(x.pg, x.pg->add_function<Square>({x.i})); }
Expression cube(const Expression& x) { return Expression(x.pg, x.pg->add_function<Cube>({x.i})); }
Expression logistic(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogisticSigmoid>({x.i})); }
Expression rectify(const Expression& x) { return Expression(x.pg, x.pg->add_function<Rectify>({ x.i })); }
Expression exponential_linear_units(const Expression& x, cnn::real scale) { return Expression(x.pg, x.pg->add_function<ExponentialLinearUnits>({ x.i }, scale)); }
Expression hinge(const Expression& x, unsigned index, cnn::real m) { return Expression(x.pg, x.pg->add_function<Hinge>({ x.i }, index, m)); }
Expression hinge(const Expression& x, const unsigned* pindex, cnn::real m) { return Expression(x.pg, x.pg->add_function<Hinge>({x.i}, pindex, m)); }
Expression log_softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSoftmax>({x.i})); }
Expression log_softmax(const Expression& x, const std::vector<unsigned>& d) { return Expression(x.pg, x.pg->add_function<RestrictedLogSoftmax>({x.i}, d)); }
Expression softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Softmax>({x.i})); }
Expression softsign(const Expression& x) { return Expression(x.pg, x.pg->add_function<SoftSign>({x.i})); }
Expression pow(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Pow>({x.i, y.i})); }
Expression min(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Min>({x.i, y.i})); }
Expression max(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Max>({x.i, y.i})); }
Expression noise(const Expression& x, cnn::real stddev) { return Expression(x.pg, x.pg->add_function<GaussianNoise>({x.i}, stddev)); }
Expression dropout(const Expression& x, cnn::real p) { return Expression(x.pg, x.pg->add_function<Dropout>({x.i}, p)); }
Expression block_dropout(const Expression& x, cnn::real p) { return Expression(x.pg, x.pg->add_function<BlockDropout>({x.i}, p)); }

Expression reshape(const Expression& x, const Dim& d) { return Expression(x.pg, x.pg->add_function<Reshape>({x.i}, d)); }
Expression transpose(const Expression& x) { return Expression(x.pg, x.pg->add_function<Transpose>({x.i})); }

Expression trace_of_product(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<TraceOfProduct>({x.i, y.i}));}
Expression cwise_multiply(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<CwiseMultiply>({x.i, y.i}));}

Expression dot_product(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<DotProduct>({x.i, y.i})); }
Expression squared_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<SquaredEuclideanDistance>({x.i, y.i})); }
Expression huber_distance(const Expression& x, const Expression& y, cnn::real c) { return Expression(x.pg, x.pg->add_function<HuberDistance>({x.i, y.i}, c)); }
Expression l1_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<L1Distance>({x.i, y.i})); }
Expression binary_log_loss(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<BinaryLogLoss>({x.i,y.i})); }
Expression pairwise_rank_loss(const Expression& x, const Expression& y, cnn::real m) { return Expression(x.pg, x.pg->add_function<PairwiseRankLoss>({x.i, y.i}, m)); }
Expression poisson_loss(const Expression& x, unsigned y) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, y)); }
Expression poisson_loss(const Expression& x, const unsigned* py) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>({x.i}, py)); }

Expression reduce(const Expression& x) { return Expression(x.pg, x.pg->add_function<Reduce>({ x.i })); }

Expression conv1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DNarrow>({x.i, f.i})); }
Expression conv1d_wide(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DWide>({x.i, f.i})); }
Expression kmax_pooling(const Expression& x, unsigned k) { return Expression(x.pg, x.pg->add_function<KMaxPooling>({x.i}, k)); }
Expression fold_rows(const Expression& x, unsigned nrows) { return Expression(x.pg, x.pg->add_function<FoldRows>({x.i}, nrows)); }

Expression pick(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, v)); }
Expression pick(const Expression& x, unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickElement>({x.i}, pv)); }
Expression pickrange(const Expression& x, unsigned v, unsigned u) { return Expression(x.pg, x.pg->add_function<PickRange>({ x.i }, v, u)); }
Expression columnslices(const Expression& x, unsigned row, unsigned start_column, unsigned exclusive_end_column) { return Expression(x.pg, x.pg->add_function<ColumnSlices>({ x.i }, row, start_column, exclusive_end_column)); }

//Expression pickneglogsoftmax(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v)); }
//Expression pickneglogsoftmax(const Expression& x, const std::vector<unsigned> & v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>({x.i}, v)); }

Expression sum_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumColumns>({x.i})); }

Expression sum_batches(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumBatches>({x.i})); }

Expression kmh_ngram(const Expression& x, unsigned n) { return Expression(x.pg, x.pg->add_function<KMHNGram>({x.i}, n)); }

} }
