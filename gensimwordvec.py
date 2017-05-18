# -*- coding: utf-8 -*-
"""
Created on Fri May 19 00:30:23 2017

@author: anirudh
"""

import gensim, logging
import os
from gensim.models.keyedvectors import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open((os.path.join(self.dirname, fname)), encoding="utf-8"):
                yield line.split()
 
sentences = MySentences("F:/python/test123") # a memory-friendly iterator

model = gensim.models.Word2Vec(sentences, min_count=8, size=100, workers = 2)

model.save("F:/python/test123/word_vec123")
new_model = gensim.models.Word2Vec.load("F:/python/test123/word_vec123")

#test 
print(new_model.most_similar('export', topn=10))

#print(new_model.similar_by_vector(new_model["company"], topn=10))

#new_model['computer']

#v = new_model.wv.vocab

word_vectors = KeyedVectors.load_word2vec_format("F:/python/test123/word_vec123", binary=True,unicode_errors='ignore')  # C binary format



