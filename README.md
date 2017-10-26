word2vec+constraints
====================
This is an open source implementation of a modified version of word2vec's skip-gram architecture that includes similarity constraints.

Requirements
-------------

Usage
------

./word2vec_constraints -train CORPORA.txt -output EMBEDDINGS.txt -size 300 -window 5 -sample 0 -negative 5 -hs 0 -binary 0 -cbow 0 -read-simconstr CONSTRAINT.txt -lambdasim LAMBDA
