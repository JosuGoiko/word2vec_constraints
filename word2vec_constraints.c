//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
//---CLEAR---
#define MAX_CONSTR_SET 1000
#define MAX_CONSTR 1000
#define SIM 0
#define SENS 1
#define LANG 2
#define S0 0
#define S1 1
#define S2 2

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn, sim_indx, sens_indx, lang_indx;
  int sim_constrnum, sens_constrnum, lang_constrnum, *point;
  char *word, *code, codelen;
};


// Similarity/language constraints are going to be 
// stored in this structure as vocabulary indexes
/*struct simlangconstr{
  long long vocab_indx;
  long long *constr_array;
};*/

// Sense constraints are going to be 
// stored in this structure as vocabulary indexes
struct constr{
  long long vocab_indx;
  long long *constr_array;
  long long *weight_array;
};

// ----Similarity constrains
long long int sim_cn = 0;
real lambdasim = 0;
int weightsim = 0;

// ----Sense constrains
long long int sens_cn = 0;
real lambdasens = 0;
int weightsens = 0;

// ----Language constrains
long long int lang_cn = 0;
real lambdalang = 0;
int weightlang = 0;

char train_file[MAX_STRING], output_file[MAX_STRING], simconstr_file[MAX_STRING], sensconstr_file[MAX_STRING], langconstr_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 4, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

//Constrain structures
struct constr *sim;
struct constr *lang;
struct constr *sens;
long long max_constr_set;

//Output space
int outspace = S2;
real *os2;

//Loss function
real loss = 0, *loss_out, fpos = 0, fneg = 0, fconst = 0;
long long contloss = 0, loss_size = 0, max_loss_size = 10000;
char loss_file[MAX_STRING];

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}


// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab[vocab_size].sim_indx = -1;//CLEAR
  vocab[vocab_size].sim_constrnum = 0;//CLEAR
  vocab[vocab_size].sens_indx = -1;//CLEAR
  vocab[vocab_size].sens_constrnum = 0;//CLEAR
  vocab[vocab_size].lang_indx = -1;//CLEAR
  vocab[vocab_size].lang_constrnum = 0;//CLEAR
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  vocab_size = 0;
  printf("\n---Creating corpus vocabulary...\n");
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void Weighting(char *sense, long long int *weight){
    char *string_in, *tok;
    int count = 0;
    
    string_in = sense;
    
    while ((tok = strsep(&string_in, ":")) != NULL)
    {
        if (count == 0) strcpy(sense,tok);
        else *weight = (long long int) atoi(tok)+1;
        count++;
    }
    
    free(string_in);
}

void simUpdate(int constr_count, long long target, long long int *line_count, long long int *c_array, long long int *c_weight){
    int i;
    
    //Update vocabulary struct wrt sim
    vocab[target].sim_indx = (*line_count) - 1;
    vocab[target].sim_constrnum = constr_count;

    //Update similarity struct wrt vocabulary index
    //and dinamically reserve memory for effective constraints
    sim[(*line_count)-1].constr_array = calloc(constr_count, sizeof(long long));
    sim[(*line_count)-1].vocab_indx = target;
    if ((weightsim == 1)&&(constr_count > 0)){
        sim[(*line_count) - 1].weight_array = calloc(constr_count, sizeof(long long));
    }
      
    //Update effective constraints' content (indexes of constraints in vocabulary)
    for(i = 0; i < constr_count; i++){
      sim[(*line_count)-1].constr_array[i] = c_array[i];
      if ((weightsim == 1)&&(constr_count > 0)){
          sim[(*line_count) - 1].weight_array[i] = c_weight[i];
          c_weight[i] = 0;
      }
      c_array[i] = 0;
    }
}

void langUpdate(int constr_count, long long target, long long int *line_count, long long int *c_array, long long int *c_weight){
    int i;
    
    //Update vocabulary struct wrt lang
    vocab[target].lang_indx = (*line_count) - 1;
    vocab[target].lang_constrnum = constr_count - 1;

    //Update language struct wrt vocabulary index
    //and dinamically reserve memory for effective constraints
    lang[(*line_count)-1].constr_array = calloc(constr_count - 1, sizeof(long long));
    lang[(*line_count)-1].vocab_indx = target;
    if ((weightlang == 1)&&((constr_count - 1) > 1)){
        lang[(*line_count) - 1].weight_array = calloc(constr_count - 1, sizeof(long long));
    }
      
    //Update effective constraints' content (indexes of constraints in vocabulary)
    for(i = 0; i < (constr_count-1); i++){
      lang[(*line_count)-1].constr_array[i] = c_array[i];
      if ((weightlang == 1)&&((constr_count - 1) > 1)){
          lang[(*line_count) - 1].weight_array[i] = c_weight[i];
          c_weight[i] = 0;
      }
      c_array[i] = 0;
    }
}


void sensUpdate(int constr_count, long long target, long long int *line_count, long long int *c_array, long long int *c_weight){
    int i;
    
    //Update vocabulary struct wrt sens
    vocab[target].sens_indx = (*line_count) - 1;
    vocab[target].sens_constrnum = constr_count - 1;

    //Update sense struct wrt vocabulary index
    //and dinamically reserve memory for effective constraints
    //and respective weights
    sens[(*line_count) - 1].constr_array = calloc(constr_count - 1, sizeof(long long));
    sens[(*line_count) - 1].vocab_indx = target;
    if ((weightsens == 1)&&((constr_count - 1) > 1)){
        sens[(*line_count) - 1].weight_array = calloc(constr_count - 1, sizeof(long long));
    }

    //Update effective constraints' content (indexes of constraints in vocabulary)
    for(i = 0; i < (constr_count - 1); i++){
      sens[(*line_count) - 1].constr_array[i] = c_array[i];
      if ((weightsens == 1)&&((constr_count - 1) > 1)){
          sens[(*line_count) - 1].weight_array[i] = c_weight[i];
          c_weight[i] = 0;
      }
      c_array[i] = 0;
    }
}

// Reads Similarity/sense/language constrain file, line by line.
// If TARGET word appears in existing vocabulary, it updates
// vocabulary structure with constrains as indexes
void ReadConstr(long long int *cn, FILE *fco, char *constr_file, int constr_type) {
  int a = 0, ch, constr_cn = 0, ch_cn = 0, flag_vocab = 1, flag_constr = 1, first = 1;
  long long itarget = -999, iconstr = -999, constr_array[MAX_CONSTR] = {0}, weight = 0, constr_weight[MAX_CONSTR] = {0};
  char constr_word[MAX_STRING], constr_line[(MAX_CONSTR*MAX_STRING)+MAX_CONSTR-1];
  
  fco = fopen(constr_file, "r");
  if (fco == NULL) {
    printf("ERROR: Constrain data file not found!\n");
    exit(1);
  }
   max_constr_set = MAX_CONSTR_SET;
  *cn = 0;
  while (fgets(constr_line, (MAX_CONSTR*MAX_STRING)+MAX_CONSTR-1, fco)) {//begin while0
    //Read constrain file, line by line
    //First word is the target word, and the rest are its constrains
    while(1){//begin while1
      ch=constr_line[ch_cn];
      ch_cn++;
      if ((ch == ' ') || (ch == '\n')){//begin if0
          if(first == 1){//begin if1
 	    //Check if the TARGET word EXISTS in the corpus vocabulary
	    itarget = SearchVocab(constr_word);
	    a = 0;
	    if (itarget == -1){
	    //Target word is not in the corpus vocabulary
	    //Check the next contrain set
	      memset(&constr_word[0], 0, MAX_STRING*sizeof(char));
	      memset(&constr_line[0], 0, sizeof(constr_line));
	      //Current constrain does not appear in corpus vocabulary
	      flag_vocab = 0;						
	      break;
	    }else{
              //Target exists in corpus vocabulary
	      if(ch == ' '){
                first = 0;
                memset(&constr_word[0], 0, MAX_STRING*sizeof(char));
                continue;
	      }else{
                //Targets without constrains are not taken into account
	        memset(&constr_line[0], 0, sizeof(constr_line));
	        memset(&constr_word[0], 0, MAX_STRING*sizeof(char));
	        continue;
	      }
	    }
	  }//end if1
          if(first == 0){
            if (((weightsens == 1)&&(constr_type == SENS))||((weightlang == 1)&&(constr_type == LANG))||((weightsim == 1)&&(constr_type == SIM))){
                Weighting(constr_word, &weight);  
            }
	    iconstr = SearchVocab(constr_word);
	    a = 0;
	    if(iconstr == -1){
	    //WordNet constrain is not in the corpus vocabulary
	    //Reset variables and check next constrain
	      memset(&constr_word[0], 0, MAX_STRING*sizeof(char));
	      if (ch != '\n')continue;
	      else break;
	    }else{
                if (constr_cn == 0){ //At least one effective constrain. Update size
                    (*cn)++;
                    if ((*cn)+2 > max_constr_set){
                          max_constr_set += MAX_CONSTR_SET;
                          if(constr_type == SIM) sim = (struct constr *)realloc(sim, max_constr_set*sizeof(struct constr));
                          if(constr_type == SENS) sens = (struct constr *)realloc(sens, max_constr_set*sizeof(struct constr));
                          if(constr_type == LANG) lang = (struct constr *)realloc(lang, max_constr_set*sizeof(struct constr));
                    }
                }
              //Constraint exists in the corpus vocabulary
	      constr_array[constr_cn] = iconstr;
              if (((weightsens == 1)&&(constr_type == SENS))||((weightlang == 1)&&(constr_type == LANG))||((weightsim == 1)&&(constr_type == SIM))){
                  constr_weight[constr_cn] = weight;
              }
	      memset(&constr_word[0], 0, MAX_STRING*sizeof(char));
	      constr_cn++;
	      //if ((ch == '\n')||(constr_cn >= MAX_CONSTR)) break;
              if (ch == '\n') break;
	    }
	  }
       //end if0
       }else{
         //Keep on reading characters until end of word
	 if(a < MAX_STRING){ constr_word[a] = ch; a++; }
	 else a = 0;
       }
    }//end while1
    //Reset character variables
    ch_cn = 0;
    ch = 0;
    a = 0;
    //Once the constraint line is read, update 
    //constraint structure size and content
    if((flag_vocab == 1) && (constr_cn > 0)){
        switch(constr_type){
            case SIM:  //SIMILARITY
                simUpdate(constr_cn, itarget, cn, constr_array, constr_weight);
                break;
            case SENS:  //SENSE
                sensUpdate(constr_cn, itarget, cn, constr_array, constr_weight);
                break;
            case LANG: //LANGUAGE
                langUpdate(constr_cn, itarget, cn, constr_array, constr_weight);
                break;
        }
    }
    //Reset variables for next constrain set
    constr_cn = 0;
    itarget = -999;
    iconstr = -999;
    flag_vocab = 1;
    flag_constr = 1;
    first = 1;
    memset(constr_array, 0, MAX_CONSTR*sizeof(long long));
    memset(constr_weight, 0, MAX_CONSTR*sizeof(long long));
    memset(&constr_line[0], 0, sizeof(constr_line));
  }//end while0
  fclose(fco);

  if (debug_mode > 0) printf("Constrained types in train file: %lld\n", *cn);

}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
 
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
}

//Accumulate gradient for constrain BP in syn0
//and perform backpropagation in SIMILARITY CONSTRAINS
//from space syn1neg
void BackPropSim(real *grad, long long target){
  long long c = 0, s = 0, s1 = 0, weight_tot = 0;
  real relative_weight = 0;
    //Reset gradient for target word
    for (c = 0; c < layer1_size; c++) grad[c] = 0;  
    weight_tot = 0;
    //Smoothing
   // if ((weightsim == 1)&&(vocab[target].sim_constrnum > 1)){
        //printf("Target: %s--\n", vocab[target].word);
   //     for (s = 0; s < vocab[target].sim_constrnum; s++){ 
   //         weight_tot += sim[vocab[target].sim_indx].weight_array[s];
   //     }
   // }
    
    //Apply every language constraint
    for (s = 0; s < vocab[target].sim_constrnum; s++){
      s1 = sim[vocab[target].sim_indx].constr_array[s] * layer1_size;
      if (weightsim == 1){
          if(vocab[target].sim_constrnum>1){ 
            relative_weight = (real)sim[vocab[target].sim_indx].weight_array[s];///weight_tot;
          }
      }else{
          relative_weight = 1;
      }
      for (c = 0; c < layer1_size; c++){
       //Accumulate gradient for target word
        grad[c] += 2*lambdasim*(syn1neg[c + target * layer1_size]- relative_weight * syn1neg[c + s1]);
        //UPDATE syn1neg's SIMILARITY CONSTRAINs with gradient -> ws = ws + nu*dJ/d(ws)
        syn1neg[c+s1] += alpha * 2 * lambdasim * relative_weight * (syn1neg[c + target * layer1_size]- relative_weight * syn1neg[c + s1]);
        if (debug_mode > 1) fconst += lambdasim*(syn1neg[c+target*layer1_size]-syn1neg[c+s1])*(syn1neg[c+target*layer1_size]-syn1neg[c+s1]);
      }
    }
}

//Accumulate gradient for constrain BP in syn0
//and perform backpropagation in LANGUAGE CONSTRAINS
//from space syn1neg
void BackPropLang(real *grad, long long target){
  long long c = 0, s = 0, s1 = 0, weight_tot = 0;
  real relative_weight = 0;
    //Reset gradient for target word
    for (c = 0; c < layer1_size; c++) grad[c] = 0;  
    weight_tot = 0;
    //Smoothing
    //if ((weightlang == 1)&&(vocab[target].lang_constrnum > 1)){
        //printf("Target: %s--\n", vocab[target].word);
    //    for (s = 0; s < vocab[target].lang_constrnum; s++){ 
    //        weight_tot += lang[vocab[target].lang_indx].weight_array[s];
    //    }
    //}
    
    //Apply every language constraint
    for (s = 0; s < vocab[target].lang_constrnum; s++){
      s1 = lang[vocab[target].lang_indx].constr_array[s] * layer1_size;
      if (weightlang == 1){
          if(vocab[target].lang_constrnum>1){ 
            relative_weight = (real)lang[vocab[target].lang_indx].weight_array[s];///weight_tot;
          }
      }else{
          relative_weight = 1;
      }
      for (c = 0; c < layer1_size; c++){
       //Accumulate gradient for target word
        grad[c] += 2*lambdalang*(syn1neg[c + target * layer1_size]- relative_weight * syn1neg[c + s1]);
        //UPDATE syn1neg's SIMILARITY CONSTRAINs with gradient -> ws = ws + nu*dJ/d(ws)
        syn1neg[c+s1] += alpha * 2 * lambdalang * relative_weight * (syn1neg[c + target * layer1_size]- relative_weight * syn1neg[c + s1]);
        if (debug_mode > 1) fconst += lambdalang*(syn1neg[c+target*layer1_size]-syn1neg[c+s1])*(syn1neg[c+target*layer1_size]-syn1neg[c+s1]);
      }
    }
}

//Accumulate gradient for constrain BP in syn0
//and perform backpropagation in SENSE CONSTRAINS
//from space syn1neg
void BackPropSens(real *grad, long long target){
  long long c = 0, s = 0, s1 = 0, weight_tot = 0;
  real relative_weight = 0;
    //Reset gradient for target word
    for (c = 0; c < layer1_size; c++) grad[c] = 0;  
    weight_tot = 0;
    //Smoothing
    //if ((weightsens == 1)&&(vocab[target].sens_constrnum > 1)){
        //printf("Target: %s--\n", vocab[target].word);
    //    for (s = 0; s < vocab[target].sens_constrnum; s++){ 
    //        weight_tot += sens[vocab[target].sens_indx].weight_array[s];
    //    }
    //}
    
    //Apply every synset constraints
    for (s = 0; s < vocab[target].sens_constrnum; s++){
      s1 = sens[vocab[target].sens_indx].constr_array[s] * layer1_size;
      if (weightsens == 1){
          if(vocab[target].sens_constrnum>1){ 
            relative_weight = (real)sens[vocab[target].sens_indx].weight_array[s];///weight_tot;
          }
      }else{
          relative_weight = 1;
      }
      //printf("%lld - %s -> %f = %lld/%lld\n",s,vocab[sens[vocab[target].sens_indx].constr_array[s]].word, weight, sens[vocab[target].sens_indx].weight_array[s],weight_tot);
      for (c = 0; c < layer1_size; c++){
      //Accumulate gradient for target word
        grad[c] += 2*lambdasens*(syn1neg[c + target * layer1_size]- relative_weight * syn1neg[c + s1]);
        //UPDATE syn1neg's SENSE CONSTRAINTs with gradient -> ws = ws + nu*dJ/d(ws)
        syn1neg[c+s1] += alpha * 2 * lambdasens * relative_weight*(syn1neg[c + target * layer1_size]- relative_weight * syn1neg[c + s1]);
        if (debug_mode > 1) fconst += lambdasens*(syn1neg[c+target*layer1_size]-syn1neg[c+s1])*(syn1neg[c+target*layer1_size]-syn1neg[c+s1]);
      }
    }
}

//Perform backpropagation in TARGET word from space syn1neg
void BackPropTarget(long long constr_indx, real *grad, long long target_indx){	
  long long int c;
  if(constr_indx != -1) for (c = 0; c < layer1_size; c++) 
       syn1neg[c + target_indx*layer1_size] -= alpha * grad[c];
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g, sigm;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  real *gsim = (real *)calloc(layer1_size, sizeof(real));    //SIMILARITY gradient
  real *gsens = (real *)calloc(layer1_size, sizeof(real));   //SENSE gradient
  real *glang = (real *)calloc(layer1_size, sizeof(real));   //LANGUAGE gradient
  FILE *fi = fopen(train_file, "rb");

  //Loss variable initialization
  if(loss_file[0] != 0)	loss_out = (real *)calloc(max_loss_size, sizeof(real));

  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        //average of accumulated loss
        loss = loss/(real)contloss;
	if(loss_file[0] != 0){
		if((loss_size+1000) > max_loss_size){
			max_loss_size += 10000;
			loss_out = (real *) realloc (loss_out, max_loss_size*sizeof(real));
		}
		loss_out[loss_size] = fabsf(loss);
		loss_size++;
	}
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk Loss: %.2f ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000),
         fabsf(loss));
        fflush(stdout);
        //reset total loss
        loss = 0;  
	contloss = 0;
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0)break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    if (lambdasim > 0) for (c = 0; c < layer1_size; c++) gsim[c] = 0;
    if (lambdasens > 0) for (c = 0; c < layer1_size; c++) gsens[c] = 0;
    if (lambdalang > 0) for (c = 0; c < layer1_size; c++) glang[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    //SKIPGRAM
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
	c = sentence_position - window + a;
	if (c < 0) continue;
	if (c >= sentence_length) continue;
	last_word = sen[c];
	if (last_word == -1) continue;
	l1 = last_word * layer1_size;
	for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
	if (negative > 0) for (d = 0; d < negative + 1; d++) {
	  if (d == 0) {
	    target = word;
	    label = 1;
	  } else {
	    next_random = next_random * (unsigned long long)25214903917 + 11;
	    target = table[(next_random >> 16) % table_size];
	    if (target == 0) target = next_random % (vocab_size - 1) + 1;
	    if (target == word) continue;
	    label = 0;
	  }
	  l2 = target * layer1_size;
	  f = 0;
	  for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
	  if (f > MAX_EXP) g = (label - 1) * alpha;
	  else if (f < -MAX_EXP) g = (label - 0) * alpha;
	  else{
              sigm = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
              g = (label - sigm) * alpha;
              //loss calculation
              if (debug_mode > 1){ 
		if(d == 0) fpos = log(sigm);
                else fneg += log(1-sigm);
	      }
          }
	  for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
	  /////Constrain backpropagation/////
	  if ((d == 0) && (a==b)){
            if ((lambdasim > 0)&&(vocab[word].sim_indx != -1)) BackPropSim(gsim, target);
            if ((lambdasens > 0)&&(vocab[word].sens_indx != -1))BackPropSens(gsens, target);
            if ((lambdalang > 0)&&(vocab[word].lang_indx != -1))BackPropLang(glang, target);
	  }
	}
	 // Learn weights input -> hidden
	for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];

	//Loss function calculation for every context word in window
	loss += fpos + fneg;
        fpos = 0;
        fneg = 0;
    }
    
    //UPDATE syn1neg's TARGET WORD with accumulated gradient -> w = w + nu*dJ/d(w)
    if (lambdasim > 0) BackPropTarget(vocab[word].sim_indx, gsim, word);
    if (lambdasens > 0) BackPropTarget(vocab[word].sens_indx, gsens, word);
    if (lambdalang > 0) BackPropTarget(vocab[word].lang_indx, glang, word);

    //Actualization of negative sampling loss function with constrains' losses
    loss -= fconst;
    fconst = 0;
    contloss++;
    
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }

  fclose(fi);
  free(neu1);
  free(neu1e);
  free(gsim);
  free(gsens);
  free(glang);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo, *fsim, *fsens, *flang, *floss;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0){
	ReadVocab();
  }else{
	LearnVocabFromTrainFile();
        if (lambdasim > 0){
          if (simconstr_file[0] != 0){
            printf("\n---Creating similarity constrains...\n");
            ReadConstr(&sim_cn, fsim, simconstr_file, SIM);
	  }
	}
        if (lambdasens > 0){
          if (sensconstr_file[0] != 0){
            printf("\n---Creating sense constrains...\n");
            ReadConstr(&sens_cn, fsens, sensconstr_file, SENS);
	  }
	}
	if (lambdalang > 0){
          if (langconstr_file[0] != 0){
            printf("\n---Creating language constrains...\n");
            ReadConstr(&lang_cn, flang, langconstr_file, LANG);
	  }
	}
  }
  if (save_vocab_file[0] != 0) SaveVocab();
  
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();

  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save loss in a file
    if(loss_file[0] != 0){
	floss = fopen(loss_file, "wb");
	printf("\n loss size %lld\n", loss_size);
	for (a = 0; a < loss_size; a++) fprintf(floss, "%lf\n", loss_out[a]);
	fclose(floss);
    }
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary){
          switch (outspace){
              case S0:
                  for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
                  break;
              case S1:
                  for (b = 0; b < layer1_size; b++) fwrite(&syn1neg[a * layer1_size + b], sizeof(real), 1, fo);
                  break;
              case S2:
                  os2 = (real *)calloc(layer1_size, sizeof(real));
                  for (b = 0; b < layer1_size; b++){
                      os2[b] = syn1neg[a * layer1_size + b]+syn0[a * layer1_size + b];
                      fwrite(&os2[b], sizeof(real), 1, fo);
                  }
                  break;
          }
      }else{
          switch (outspace){
              case S0:
                  for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
                  break;
              case S1:
                  for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]);
                  break;
              case S2:
                  for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1neg[a * layer1_size + b]+syn0[a * layer1_size + b]);
                  break;
          }
            
      }
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t\tThe loss will be saved to <file>\n");
    printf("\t-loss <file>\n");
    printf("\t-read-simconstr <file>\n");
    printf("\t\tThe similarity constrains will be read from <file>\n");
    printf("\t-lambdasim <float>\n");
    printf("\t\tWeight for similarity constrains' regularizer\n");
    printf("\t-lambdalang <float>\n");
    printf("\t\tWeight for language constrains' regularizer\n");
    printf("\t-lambdasens <float>\n");
    printf("\t\tWeight for sense constrains' regularizer\n");
    printf("\t-weightsens <float>\n");
    printf("\t\tWeighting in sense constraints: default is 0 (off)\n");
    printf("\t-out-space <int>\n");
    printf("\t\tOutput vector space; default is 2 for syn0+syn1neg (use 0 for syn0, 1 for syn1neg)\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  simconstr_file[0] = 0;
  langconstr_file[0] = 0;
  sensconstr_file[0] = 0;
  loss_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-simconstr", argc, argv)) > 0) strcpy(simconstr_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-langconstr", argc, argv)) > 0) strcpy(langconstr_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-sensconstr", argc, argv)) > 0) strcpy(sensconstr_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-lambdasim", argc, argv)) > 0) lambdasim = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambdalang", argc, argv)) > 0) lambdalang = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambdasens", argc, argv)) > 0) lambdasens = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-weightsens", argc, argv)) > 0) weightsens = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-weightsim", argc, argv)) > 0) weightsim = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-weightlang", argc, argv)) > 0) weightlang = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-out-space", argc, argv)) > 0) outspace = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-loss", argc, argv)) > 0) strcpy(loss_file, argv[i + 1]);  
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  sim = (struct constr *)calloc(MAX_CONSTR_SET, sizeof(struct constr));
  lang = (struct constr *)calloc(MAX_CONSTR_SET, sizeof(struct constr));
  sens = (struct constr *)calloc(MAX_CONSTR_SET, sizeof(struct constr));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  free(sim);
  free(sens);
  free(lang);
  free(os2);
  free(loss_out);
  return 0;
}
