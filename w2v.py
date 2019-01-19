#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]
 
    model = Word2Vec(LineSentence(inp), size=100, window=10, min_count=1,sg=1,hs=1,
                     workers=multiprocessing.cpu_count())
    #window:skip-gramͨ����10������CBOWͨ����5����
    #hs: ���Ϊ1������hierarchica softmax���ɡ��������Ϊ0��defaut������negative sampling�ᱻʹ�á�
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)