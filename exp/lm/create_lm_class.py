#!/usr/bin/env python

import sys
import os
import re

from lm import *


turn = 0
operator_sys = ''


if __name__ == "__main__":
        assert len(sys.argv) == 5, "Usage: %s <lm data> <nbr class> <wrd2cls file> <cls2wrd file>" % sys.argv[0]


        construct_word_class(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
        
# real run
# python create_lm_class.py c:/data/ptbdata/ptb.trn 100 c:/data/ptbdata/wrd2cls.txt c:/data/ptbdata/cls2wrd.txt
