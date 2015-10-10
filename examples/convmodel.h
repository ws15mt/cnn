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
#include "expr-xtra.h"

#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>

namespace cnn {

    struct ConvLayer {
        ConvLayer() {};

        // in_rows = rows per word in input matrix
        // k_fold_rows = 1 no folding, 2 fold two rows together, 3 ...
        // filter_width = length of filter (columns)
        // in_nfmaps = number of feature maps in input
        // out_nfmaps = number of feature maps in output
        ConvLayer(Model&m, int in_rows, int k_fold_rows, int filter_width, int in_nfmaps, int out_nfmaps) :
            p_filts(in_nfmaps),
            p_fbias(in_nfmaps),
            k_fold_rows(k_fold_rows) {
            if (k_fold_rows < 1 || ((in_rows / k_fold_rows) * k_fold_rows != in_rows)) {
                cerr << "Bad k_fold_rows=" << k_fold_rows << endl;
                abort();
            }
            for (int i = 0; i < in_nfmaps; ++i) {
                p_filts[i].resize(out_nfmaps);
                p_fbias[i].resize(out_nfmaps);
                for (int j = 0; j < out_nfmaps; ++j) {
                    p_filts[i][j] = m.add_parameters({ in_rows, filter_width }, 0.01);
                    p_fbias[i][j] = m.add_parameters({ in_rows }, 0.05);
                }
            }
            //for (int j = 0; j < out_nfmaps; ++j)
            //p_fbias[j] = m.add_parameters({in_rows});
        }

        vector<Expression> apply(ComputationGraph& cg, const vector<Expression>& inlayer, int k_out) const {
            const unsigned out_nfmaps = p_filts.front().size();
            const unsigned in_nfmaps = p_filts.size();
            if (in_nfmaps != inlayer.size()) {
                cerr << "Mismatched number of input features (" << inlayer.size() << "), expected " << in_nfmaps << endl;
                abort();
            }
            vector<Expression> r(out_nfmaps);

            vector<Expression> tmp(in_nfmaps);
            for (unsigned int fj = 0; fj < out_nfmaps; ++fj) {
                for (unsigned int fi = 0; fi < in_nfmaps; ++fi) {
                    Expression t = conv1d_wide(inlayer[fi], parameter(cg, p_filts[fi][fj]));
                    t = colwise_add(t, parameter(cg, p_fbias[fi][fj]));
                    tmp[fi] = t;
                }
                Expression s = sum(tmp);
                if (k_fold_rows > 1)
                    s = fold_rows(s, k_fold_rows);
                s = kmax_pooling(s, k_out);
                r[fj] = rectify(s);
            }
            return r;
        }
        vector<vector<Parameters*>> p_filts; // [feature map index from][feature map index to]
        vector<vector<Parameters*>> p_fbias; // [feature map index from][feature map index to]
        int k_fold_rows;
    };

    template<class TConvLayer>
    struct ConvNet {
        unsigned nInputDim, nOutputDim;
        LookupParameters* p_w;
        TConvLayer cl1;
        TConvLayer cl2;
        Parameters* p_t2o;
        Parameters* p_obias;

        explicit ConvNet(Model& m, unsigned vocab_size, unsigned input_dim, unsigned output_dim) :
            p_w(m.add_lookup_parameters(vocab_size, { long(input_dim) })),
            //ConvLayer(Model&m, int in_rows, int k_fold_rows, int filter_width, int in_nfmaps, int out_nfmaps) :
            cl1(m, input_dim, 2, 10, 1, 6),
            cl2(m, input_dim / 2, 2, 6, 6, 14),
            p_t2o(m.add_parameters({ long(output_dim), 14 * (long(input_dim) / 4) * 5 })),
            p_obias(m.add_parameters({ long(output_dim) })),
            nInputDim(input_dim),
            nOutputDim(output_dim)
        {
        }

        Expression BuildClassifier(const vector<int>& x, ComputationGraph& cg, bool for_training) {
            Expression t2o = parameter(cg, p_t2o);
            Expression obias = parameter(cg, p_obias);
            int k_2 = 5;
            int len = x.size();
            int k_1 = max(k_2, len / 2);
            vector<Expression> vx(x.size());
            for (unsigned i = 0; i < x.size(); ++i)
                vx[i] = lookup(cg, p_w, x[i]);
            Expression s = concatenate_cols(vx);

            vector<Expression> l0(1, s);
            vector<Expression> l1 = cl1.apply(cg, l0, k_1);
            vector<Expression> l2 = cl2.apply(cg, l1, k_2);
            for (auto& fm : l2)
                fm = reshape(fm, { long(k_2 * nInputDim / 4) });
            Expression t = concatenate(l2);

            Expression r = t2o * t + obias;
            return r;
        }
    };

}; // namespace cnn
