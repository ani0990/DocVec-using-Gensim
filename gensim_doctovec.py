import gensim , logging
import os
import collections
import smart_open
import random


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))


############ Model Train ################

"""
model = gensim.models.doc2vec.Doc2Vec(size=50, window = 50, min_count=2, iter=55)

model.build_vocab(train_corpus)
for epoch in range(5):
    model.train(train_corpus,total_examples=len(train_corpus),epochs=100)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    print("Epoch %s completed...." %(epoch))

model.save("gensim_model.model")

"""

############ Model Test ################

model_loaded = gensim.models.doc2vec.Doc2Vec.load("gensim_model.model")


# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus))
inferred_vector = model_loaded.infer_vector(test_corpus[doc_id])
sims = model_loaded.docvecs.most_similar([inferred_vector])
print(' '.join(test_corpus[doc_id]))
print("\n")
# print(model_loaded.docvecs.most_similar("hello there"))
# docvec = model_loaded.infer_vector(test_corpus[0])
# print(docvec)

for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: %s\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
