library(pacman); p_load(reticulate); repl_python()

# UTILS ------------------------------------------------------------------------

import pandas as pd
from tqdm import tqdm 
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from constants import *
from math import log
from numpy import array
from numpy import argmax


def infer_sentences(model, sentences, start):
    """

    Args:
        model (POSTagger): model used for inference
        sentences (list[str]): list of sentences to infer by single process
        start (int): index of first sentence in sentences in the original list of sentences

    Returns:
        dict: index, predicted tags for each sentence in sentences
    """
    res = {}
    for i in range(len(sentences)):
        res[start+i] = model.inference(sentences[i])
    return res
    
def compute_prob(model, sentences, tags, start):
    """

    Args:
        model (POSTagger): model used for inference
        sentences (list[str]): list of sentences 
        sentences (list[str]): list of tags
        start (int): index of first sentence in sentences in the original list of sentences


    Returns:
        dict: index, probability for each sentence,tag pair
    """
    res = {}
    for i in range(len(sentences)):
        res[start+i] = model.sequence_probability(sentences[i], tags[i])
    return res
    

#from https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list    
def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    df_sentences = pd.read_csv(open(sentence_file))
    doc_start_indexes = df_sentences.index[df_sentences['word'] == '-DOCSTART-'].tolist()
    num_sentences = len(doc_start_indexes)

    sentences = [] # each sentence is a list of tuples (index,word)
    if tag_file:
        df_tags = pd.read_csv(open(tag_file))
        tags = []
    for i in tqdm(range(num_sentences)):
        index = doc_start_indexes[i]
        if i == num_sentences-1:
            # handle last sentence
            next_index = len(df_sentences)
        else:
            next_index = doc_start_indexes[i+1]

        sent = []
        tag = []
        for j in range(index, next_index):
            word = df_sentences['word'][j].strip()
            if not CAPITALIZATION or word == '-DOCSTART-':
                word = word.lower()
            sent.append(word)
            if tag_file:
                tag.append((df_tags['tag'][j]))
        if STOP_WORD:
            sent.append('<STOP>')
        sentences.append(sent)
        if tag_file: 
            if STOP_WORD:           
                tag.append('<STOP>')
            tags.append(tag)

    if tag_file:
        return sentences, tags

    return sentences

def confusion_matrix(tag2idx,idx2tag, pred, gt, fname):
    """Saves the confusion matrix

    Args:
        tag2idx (dict): tag to index dictionary
        idx2tag (dict): index to tag dictionary
        pred (list[list[str]]): list of predicted tags
        gt (_type_): _description_
        fname (str): filename to save confusion matrix

    """
    matrix = np.zeros((len(tag2idx), len(tag2idx))) #-2 for start/end states 
    flat_pred = []
    flat_y = []
    for p in pred:
        flat_pred.extend(p)
    for true in gt:
        flat_y.extend(true)
    for i in range(len(flat_pred)):
        idx_pred = tag2idx[flat_pred[i]]
        idx_y = tag2idx[flat_y[i]]
        matrix[idx_y][idx_pred] += 1
    df_cm = pd.DataFrame(matrix, index = [idx2tag[i] for i in range(len(tag2idx))],
                columns = [idx2tag[i] for i in range(len(tag2idx))])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=False)
    plt.savefig(fname)

# def viterbi(y, A, B, Pi=None):
#     """
#     Return the MAP estimate of state trajectory of Hidden Markov Model.
# 
#     Parameters
#     ----------
#     y : array (T,)
#         Observation state sequence. int dtype.
#     A : array (K, K)
#         State transition matrix. See HiddenMarkovModel.state_transition  for
#         details.
#     B : array (K, M)
#         Emission matrix. See HiddenMarkovModel.emission for details.
#     Pi: optional, (K,)
#         Initial state probabilities: Pi[i] is the probability x[0] == i. If
#         None, uniform initial distribution is assumed (Pi[:] == 1/K).
# 
#     Returns
#     -------
#     x : array (T,)
#         Maximum a posteriori probability estimate of hidden state trajectory,
#         conditioned on observation sequence y under the model parameters A, B,
#         Pi.
#     T1: array (K, T)
#         the probability of the most likely path so far
#     T2: array (K, T)
#         the x_j-1 of the most likely path so far
#     """
#     # Turn (47,47) diagonal array into (47,)
#     # y = y.sum(axis = 1, keepdims=True).squeeze()
#     # Cardinality of the state space
#     K = A.shape[0]
#     # Initialize the priors with default (uniform dist) if not given by caller
#     Pi = Pi if Pi is not None else np.full(K, 1 / K)
#     T = len(y)
#     T1 = np.empty((K, T), 'd')
#     T2 = np.empty((K, T), 'B')
# 
#     # Initilaize the tracking tables from first observation
#     T1[:, 0] = Pi * B[:, y[0]]
#     T2[:, 0] = 0
# 
#     # Iterate throught the observations updating the tracking tables
#     for i in range(1, T):
#         T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
#         T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)
# 
#     # Build the output, optimal model trajectory
#     x = np.empty(T, 'B')
#     x[-1] = np.argmax(T1[:, T - 1])
#     for i in reversed(range(1, T)):
#         x[i - 1] = T2[x[i], i]
#     print([T2.argmax(), T1.argmax()])
#     return x, T1, T2

def get_index(tag1, tag2, tag2idx):
    idx1 = tag2idx[tag1]
    idx2 = tag2idx[tag2]
    n = len(tag2idx.keys()) 
    return idx1*n + idx2
    
def viterbi(y, A, B, tag2idx, idx2word,Pi = None):
    # y : array (T,)
    #     Observation state sequence. int dtype.
    # A : array (K, K, K )
    #     State transition matrix. See HiddenMarkovModel.state_transition  for
    #     details.
    # B : array (K, M)
    #     Emission matrix. See HiddenMarkovModel.emission for details.
    # Pi: optional, (K,)
    #     Initial state probabilities: Pi[i] is the probability x[0] == i. If
    #     None, uniform initial distribution is assumed (Pi[:] == 1/K).
    possible_states = A.shape[0]
    N = A.shape[0]
    length_of_sentence = len(y)
    v = np.zeros((possible_states,possible_states,length_of_sentence))
    bp = np.zeros((possible_states,possible_states, length_of_sentence))
    v[tag2idx["O"],tag2idx["O"],0] = 1
    has_values = [(tag2idx["O"],tag2idx["O"])]
    print(tag2idx)
    for i in range(1, length_of_sentence):
        new_has_values = []
        for value1,value2 in has_values:
            emissions = B[:,y[i]]

            transition = A[:,value2,:]
            # getting all memorized values in the form x,value2
            prev = v[:,value2, i-1] 
            for n,x in enumerate(prev):
                transition[n] = transition[n]* x
            v[value2,:,i] = emissions * np.max(transition,0)
            bp[:,value2,i] = value1
            mean = np.mean(emissions * np.max(transition,0))
            print(emissions * np.max(transition,0))
            for prob_i,prob in enumerate(emissions * np.max(transition,0)):
                if prob/mean > 1:
                    new_has_values.append((value2,prob_i))
        print(idx2word[y[i]], has_values)
        has_values = new_has_values
    ret = np.zeros(length_of_sentence)
    ret[-1] = tag2idx["<STOP>"]
    curr = np.argmax(v[:,tag2idx["<STOP>"],length_of_sentence-1])
    for i in reversed(range(1,length_of_sentence)):
        ret[i-1] = int(curr)
        curr = int(bp[0,curr,i])
    return ret
    

def deleted_interpolation(unigram_c, bigram_c, trigram_c):
    lambda1 = 0
    lambda2 = 0
    lambda3 = 0
        
    for a in range(len(pos_tagger.trigrams)):
        for b in range(len(pos_tagger.trigrams)):
            for c in range(len(pos_tagger.trigrams)):
                v = pos_tagger.trigrams[(a, b, c)]
                if v > 0:
                    try:
                        c1 = float( v-1 ) / ( pos_tagger.bigrams[(a, b)]-1 )
                    except ZeroDivisionError:
                        c1 = 0
                    try:
                        c2 = 
                    except ZeroDivisionError:
                        c2 = 0
                    try:
                        c3 = float( pos_tagger.unigrams.sum(axis = 1)[(a,)]-1 ) / pos_tagger.unigrams.sum(axis = 1).sum(axis = 0) - 1 
                    except ZeroDivisionError:
                        c3 = 0
         
                    k = np.argmax([c1, c2, c3])
                    if k == 0:
                        lambda3 += v
                    if k == 1:
                        lambda2 += v
                    if k == 2:
                        lambda1 += v
  
    weights = [lambda1, lambda2, lambda3]
    norm_w = [float(a)/sum(weights) for a in weights]
    return [weights, norm_w]

# Contants ---------------------------------------------------------------------

### Append stop word ###
STOP_WORD = True
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

### Inference Types ###
GREEDY = 0
BEAM = 1; BEAM_K = 2
VITERBI = 2
INFERENCE = VITERBI 

### Smoothing Types ###
LAPLACE = 0; LAPLACE_FACTOR = .2
INTERPOLATION = 1; LAMBDAS =  None
SMOOTHING = INTERPOLATION

### Append stop word ###
STOP_WORD = True

### Capitalization
CAPITALIZATION = True

# NGRAMM
NGRAMM = 3

## Handle unknown words TnT style
TNT_UNK = True
UNK_C = 10 #words with count to be considered
UNK_M = 10 #substring length to be considered

from multiprocessing import Pool
import numpy as np
import time
from utils import *
np.set_printoptions(suppress=True) # disable scientific notation


""" Contains the part of speech tagger class. """


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    # unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")


    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    # unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    # print("Unk token acc: {}".format(unk_token_acc))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        
        pass
    
    
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        self.unigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        for sentence_idx in range(len(self.data[0])):   
            sentence = self.data[1][sentence_idx]     
            for word_idx in range(len(sentence) - 0):
                pos_first = sentence[word_idx]
                pos_second = sentence[word_idx + 0]
                idx_first = self.tag2idx[pos_first]
                idx_second = self.tag2idx[pos_second]
                self.unigrams[idx_first,idx_second] += 1
        # Do we need to normalize this by dividing each instance by sum of all cases?????
        # self.unigrams = self.unigrams / self.unigrams.sum(axis = 1).sum(axis = 0)

        # office hours: take log of probabilities
        pass

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        self.bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        for sentence_idx in range(len(self.data[0])):   
            sentence = self.data[1][sentence_idx]     
            for word_idx in range(len(sentence) - 1):
                pos_first = sentence[word_idx]
                pos_second = sentence[word_idx + 1]
                idx_first = self.tag2idx[pos_first] 
                idx_second = self.tag2idx[pos_second] 
                self.bigrams[idx_first,idx_second] += 1
        # Smoothing: add 0.00001 to cells
        k = .000001 # try big range of values. .001, 2, 5 
        self.bigrams = self.bigrams + k 
        # 
        # diving every row by its sum
        self.bigrams = self.bigrams / self.bigrams.sum(axis = 1, keepdims = True)
        pass

    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        self.trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        for sentence_idx in range(len(self.data[0])):   
            sentence = self.data[1][sentence_idx]
            sentence = ["O"] + sentence
            for word_idx in range(len(sentence) - 2):
                pos_first = sentence[word_idx]
                pos_second = sentence[word_idx + 1]
                pos_third = sentence[word_idx + 2]
                idx_first = self.tag2idx[pos_first]
                idx_second = self.tag2idx[pos_second]
                idx_third = self.tag2idx[pos_third]
                self.trigrams[idx_first,idx_second,idx_third] += 1
        # Smoothing: add 0.00001 to cells
        self.trigrams = self.trigrams + 0.000001   
        # diving every row by its sum
        self.trigrams = self.trigrams / self.trigrams.sum(axis = 2, keepdims = True)
        pass

    # def get_quadgrams(self):
        
    
    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        self.lexical = np.zeros((len(self.all_tags), len(self.all_words)))
        for sentence_idx in range(len(self.data[0])):
            sentence = self.data[1][sentence_idx]
            for word_idx in range(len(sentence)):
                word = self.data[0][sentence_idx][word_idx]
                tag = self.data[1][sentence_idx][word_idx]
                idx_tag = self.tag2idx[tag]
                idx_word = self.word2idx[word]
                self.lexical[idx_tag,idx_word] += 1
        # Smoothing: add 0.00001 to cells
        self.lexical = self.lexical + 0.000001
        # diving every row by its sum
        self.lexical = self.lexical / self.lexical.sum(axis = 0, keepdims = True)

        pass

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        """
        self.data = data
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}
        self.all_words = list(set([word for sentence in self.data[0] for word in sentence]))
        self.word2idx = {self.all_words[i]:i for i in range(len(self.all_words))}
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        # TODO
        pass

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition probabilities.
        """
        self.sequence = sequence    # np.array(["I", "am", "happy"])
        
        ## TODO
        return 0. 
        pass

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.
        You should implement different kinds of inference (suggested as separate
        methods):
            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        # 0 for viterbi, 1 for greedy
        flag = 1
        
        ret = []
        if flag == 0:
            idxseq = []
            for word in sequence:
                idxseq.append(self.word2idx[word])
            x = viterbi(idxseq, self.trigrams, self.lexical, self.tag2idx,self.idx2word)
            for tag in x:
                ret.append(self.idx2tag[tag])
            return ret
        # GREEDY
        if flag == 1:
            idxseq = []
            for word in sequence:
                idxseq.append(self.word2idx[word])
            data = self.lexical[:,idxseq]
            # loop through columns to get index of highest value
            pred_index = []
            for i in range( data.shape[1] ):
                pred_index.append( data[:,i].argmax() )
            # Now that we have index of predicted tag, we get that tag
            pred_tags = []
            for i in pred_index:
                pred_tags.append(self.idx2tag[i])
            # Return predicted tags
            return pred_tags



# if __name__ == "__main__":

    os.chdir("/Users/octavioelias/Documents/_fall22/_hw2/")

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")
    
    pos_tagger = POSTagger()

    pos_tagger.train(train_data)

    # Printing/Testing ----------------------------------------------
    pos_tagger.get_emissions(); pos_tagger.get_bigrams(); 
    pos_tagger.get_trigrams(); pos_tagger.get_unigrams()
    # print(pos_tagger.trigrams[0,3,:])
    #print(pos_tagger.tag2idx.keys()) 
    print(pos_tagger.word2idx["<STOP>"])
    print(pos_tagger.inference(["-docstart-","Alex","was","on","the","big","red","house",".","<STOP>"]))
    
    # print(pos_tagger.lexical)

    # pos_tagger.greedy_decoder()

    # print(
    #     pos_tagger.inference(["-docstart-","Fed","raised","interest","<STOP>"])
    #     )
    # )

    # evaluate(
    #     train_data,
    #     pos_tagger
    # )
    # beam_search_decoder(['good', 'morning'], 1)

    # print(test_data)

    # evaluate( 
    #     test_data,
    #     pos_tagger
    #  )
    #  End of Testing -----------------------------------------------   

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    # evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    # test_predictions = []
    # for sentence in test_data:
    #     test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # TODO


    #office hours: consider f score, 

    #getting uknown words, prefix and suffix, feature engineering 
    
    
    
    
    def inference(sequence):
        """Tags a sequence with part of speech tags.
        You should implement different kinds of inference (suggested as separate
        methods):
            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        # GREEDY = 0 BEAM = 1; BEAM_K = 2 VITERBI = 2
        GREEDY = 0
        flag = GREEDY
        # GREEDY
        if flag == GREEDY:
            idxseq = []
            for word in sequence:
                idxseq.append(self.word2idx[word])
            data = self.lexical[:,idxseq]
            # loop through columns to get index of highest value
            pred_index = []
            for i in range( data.shape[1] ):
                pred_index.append( data[:,i].argmax() )
            # Now that we have index of predicted tag, we get that tag
            pred_tags = []
            for i in pred_index:
                pred_tags.append(self.idx2tag[i])
            # Return predicted tags
            return pred_tags
        # VITERBI
        ret = []
        if flag == VITERBI:
            idxseq = []
            for word in sequence:
                idxseq.append(self.word2idx[word])
            x = viterbi2(idxseq, self.trigrams, self.lexical, self.tag2idx)
            for tag in x:
                ret.append(self.idx2tag[tag])
            return ret
    
      
      
def subcategorize(sequence):
    idxseq = []
    for word in sequence:
      # if word doesnt exist in train dataset
      if (word in pos_tagger.all_words) == False:
          if re.search(r'\d', word):
              print( '_NUM_' )
          elif re.search(r'(ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)',word):
              print( '_NOUNLIKE_' )
          elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem|ing\b)', word):
              print( '_VERBLIKE_' )
          elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)',word):
              print( '_ADJLIKE_' )
          else:
              print( '_RARE_' )
      





































