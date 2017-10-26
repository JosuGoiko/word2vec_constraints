word2vec+constraints
====================
This is an open source implementation of a modified version of word2vec's skip-gram architecture that includes similarity constraints.

The modified skip-gram is able to jointly learn from genuine text and a external source such as a knowledge base via a regularization term in the former loss function. This regularization term incorporates extra semantic information from the external source, e.g. synonymy-related words from a knowledge base, into the learning process. Thus, the regularizer term adds co-occurrences from a external source to the ones that appear in the corpus, creating a joint embedding space which merges semantic information from both sources.


Usage
------

´´´
./word2vec_constraints -train CORPORA.txt -output EMBEDDINGS.txt -size 300 -window 5 -sample 0 -negative 5 -hs 0 -binary 0 -cbow 0 -read-simconstr CONSTRAINT.txt -lambdasim LAMBDA
´´´
