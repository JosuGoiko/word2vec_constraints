word2vec+constraints
====================
This is an open source implementation of a modified version of word2vec's skip-gram architecture that includes similarity constraints.

The modified skip-gram is able to jointly learn from genuine text and a external source such as a knowledge base via a regularization term in the former loss function. This regularization term incorporates extra semantic information from the external source, e.g. synonymy-related words from a knowledge base, into the learning process. Thus, the regularizer term adds co-occurrences that do not appear in the corpus, creating a joint embedding space which merges semantic information from both sources.

The algorithm is prepared to include up to three external sources at once, allowing to enrich the information in the corpus with semantic information of different nature. 

Data you need
---------------

1. Monolingual or multilingual corpus
2. Monolingual or multilingual constraint file

Each line in the constraint file should start with a target word, followed by its corresponding contraints (space delimited). The following example shows the English-Basque bilingual synonymy-related constraints for the word moon in WordNet 3.0g:

moon ilargi ilargi-argi lunation moonlight moonshine ...

Usage
------

The following command shows 

```
./word2vec_constraints -train CORPORA.txt -output EMBEDDINGS.txt -size SIZE -window W -negative NG -cbow 0 -read-simconstr CONSTRAINTS.txt -lambdasim LAMBDA

./word2vec_constraints -train ENEU.txt -output ENEU.emb -size 300 -window 5 -negative 5 -cbow 0 -read-simconstr ENEU.cst -lambdasim 0.01
```
where "-read-simconstr" makes reference to the constraints file and "-lambdasim" to the regularization coefficient in the regularizer (usually, a coeffcient of 0.01 gives good results). 
