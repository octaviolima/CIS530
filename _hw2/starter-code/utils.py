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

def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Turn (47,47) diagonal array into (47,)
    # y = y.sum(axis = 1, keepdims=True).squeeze()
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    print([T2.argmax(), T1.argmax()])
    return x, T1, T2

def get_index(tag1, tag2, tag2idx):
    idx1 = tag2idx[tag1]
    idx2 = tag2idx[tag2]
    n = len(tag2idx.keys()) 
    return idx1*n + idx2
    
def viterbi(y, A, B, tag2idx,idx2tag ,idx2word,Pi = None ):
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
    # debug = 1
    possible_states = A.shape[0]
    N = A.shape[0]
    length_of_sentence = len(y)
    v = np.zeros((possible_states,possible_states,length_of_sentence))
    bp = np.zeros((possible_states,possible_states, length_of_sentence))
    v[tag2idx["O"],tag2idx["O"],0] = 1
    has_values = [(tag2idx["O"],tag2idx["O"])]
    print(A[tag2idx["IN"],tag2idx["NN"],:])
    if debug:
        print(tag2idx)
    for i in range(1, length_of_sentence):
        new_has_values = []
        possible = []
        for value1,value2 in has_values:

            emissions = B[:,y[i]]
            transition = np.copy(A[:,value2,:])
            
            # getting all memorized values in the form x,value2
            prev = v[:,value2, i-1] 
            for n,x in enumerate(prev):
                transition[n] = transition[n]* x
            
            v[value2,:,i] = emissions * np.max(transition,0)
            bp[:,value2,i] = value1
            mean = np.mean(emissions * np.max(transition,0))
            
            
            for prob_i,prob in enumerate(emissions * np.max(transition,0)):
                if prob/mean > 1:
                    new_has_values.append((value2,prob_i))
                    possible.append(prob_i)
        if debug:
            if len(has_values) >0:
                print(idx2word[y[i]], idx2tag[has_values[0][0]],idx2tag[has_values[0][1]] )
                # print(emissions * np.max(transition,0))
                
            else:
                print(idx2word[y[i]], has_values)
            # print(transition[has_values[0][0],possible[0]])
            #print((emissions * np.max(transition,0)), "\n" )
            
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
                        c2 = float( pos_tagger.bigrams[(a, b)]-1 ) / ( pos_tagger.unigrams.sum(axis = 1)[(a,)]-1 ) 
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