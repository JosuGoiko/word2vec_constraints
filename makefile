CC = gcc 
#The -Ofast might not work with older versions of gcc; in that case, use -O2
#CFLAGS = -lm -pthread -g -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec_constrains

word2vec_constrains : word2vec_constraints.c
	$(CC) word2vec_constraints.c -o word2vec_constraints $(CFLAGS)

clean:
	rm -rf word2vec_constraints
