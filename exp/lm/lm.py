#!/usr/bin/env python

import sys
import math
import numpy as np
import operator
turn = 0
operator_sys = ''

'''
return counts
'''
def counts(ifile, norder):

    ifn = open(ifile, 'rt')
    gram = {}

    for iline in ifn:
        sline = iline.strip().split(' ')

        for index, word in enumerate(sline):
            len_sline = len(sline)
            if index < len_sline - norder + 1:
                cxt = ''
                for k in xrange(0, norder):
                    cxt = cxt + sline[index + k] + ' '
                if cxt in gram :
                    gram[cxt] = gram[cxt] + 1
                else:
                    gram[cxt] = 1
            else:
                break
    ifn.close()
    return gram

def org_and_lower_counts(ifile, norder):

    count_org_order = counts(ifile, norder)
    count_lower_order = counts(ifile, norder - 1)

    return count_org_order, count_lower_order

def prob(testfile, count_org_order, count_lower_order, norder):

    ifn = open(testfile, "rt")
    total_words = 0
    lprob  = 0.0
    for iline in ifn:
        sline = iline.strip().split(' ')
        for index, word in enumerate(sline):
            if index < len(sline) - norder +1:
                cxt = ''
                for k in xrange(0, norder):
                    cxt = cxt + sline[index + k] + ' '
                if cxt in count_org_order:
                    ncxt = count_org_order[cxt]
                else:
                    ncxt = 0

                cxt_cxt = ''
                for k in xrange(0, norder-1):
                    cxt_cxt = cxt_cxt + sline[index + k] + ' '
                if cxt_cxt in count_lower_order:
                    ncxt_cxt = count_lower_order[cxt_cxt]
                else:
                    ncxt_cxt = 0.
                if ncxt_cxt > 0. and ncxt > 0.:
                    prob = (ncxt + .0) / (ncxt_cxt + .0)
                    lprob = lprob + np.log(prob)
                else:
                    lprob = lprob - 23.
                total_words += 1
            else:
                break
    total_lprob = lprob / total_words
    ppl = math.exp(-total_lprob)
    ifn.close()

    return ppl

def construct_word_class(lmfn, nbr_cls, output_word2cls_fn, output_cls2nbrwords_fn):
    unigram = counts(lmfn, 1)
    sorted_x = sorted(unigram.items(), key = operator.itemgetter(1), reverse=True)
    n_counts = sum(unigram.itervalues(),0)
    bin_size  = n_counts / (nbr_cls + .0)
    acc_cnt = 0
    acc_cls = 0
    wrd2cls = []
    cls2nbrwrds = []
    prvwrdidx = -1
    wrdidx = 0
    for wrd, cnt in sorted_x:
        acc_cnt += cnt
        if acc_cnt > (acc_cls + 1) * bin_size:
            acc_cls += 1
            cls2nbrwrds.append((acc_cls,wrdidx - prvwrdidx))
            prvwrdidx = wrdidx
        wrd2cls.append((wrd, acc_cls))
        wrdidx += 1

    with open(output_word2cls_fn, 'wt') as ofw:
        for w,c in wrd2cls:
            print >> ofw, str(w) + '\t' + str(c)

    with open(output_cls2nbrwords_fn, 'wt') as ofw:
        for c,w in cls2nbrwrds:
            print >> ofw, str(c) + '\t' + str(w)
