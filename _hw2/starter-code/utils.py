import re
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

def viterbi2(y, T, E,word2idx, tag2idx):

    # Cardinality of the state space
    num_tags = T.shape[0]
    log_transition = np.log(T)
    log_emission = np.log(E)
    # Initialize the priors with default (uniform dist) if not given by caller
    N = len(y)
    v = np.zeros((num_tags, N))
    bp = np.zeros((num_tags, N))
    # Initilaize the tracking tables from first observation
    v[:,0] = log_emission[:,word2idx[y[0]]]
    # Iterate throught the observations updating the tracking tables
    for i in range(1, N):
        if y[i] in word2idx:
            emissions = log_emission[np.newaxis,:,word2idx[y[i]]]
        else:
            emissions = np.log(unknown_words(tag2idx, y[i]))
        prev = v[:,i-1]
        transition = log_transition
        v[:, i] = np.max(prev + transition.T + emissions.T, 1)
        bp[:, i] = np.argmax(prev + transition.T, 1)

    # Build the output, optimal model trajectory
    x = np.zeros(N)
    x[-1] = np.argmax(v[:, N - 1])
    for i in reversed(range(1, N)):
        x[i - 1] = bp[int(x[i]), i]
    return x
    
def viterbi3(y, T, E, tag2idx, word2idx, idx2tag):
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
    debug = 0
    log_transition = np.log(T)
    log_emission = np.log(E)
    possible_states = T.shape[0]
    N = T.shape[0]
    length_of_sentence = len(y)
    v = np.zeros((possible_states,possible_states,length_of_sentence))
    v += float('-inf')
    bp = np.zeros((possible_states,possible_states, length_of_sentence))
    v[tag2idx["O"],tag2idx["O"],0] = 1
    has_values = [(tag2idx["O"])]
    if debug:
        print(tag2idx)
    for i in range(1, length_of_sentence):
        new_has_values = set()
        possible = set()
        for value2 in has_values:
            if y[i] not in word2idx:
                emissions = np.log(np.array(unknown_words(tag2idx, y[i])))
                emissions = emissions[np.newaxis,:]
            else:
                emissions = log_emission[np.newaxis,:,word2idx[y[i]]]
            transition = log_transition[:,value2,:]
            prev = v[:,value2, i-1] 
            v[value2,:,i] = np.max(prev + transition.T + emissions.T, 1)
            bp[:,value2,i] = np.argmax(prev + transition.T, 1)

            if y[i] in word2idx:
                for prob_i,prob in enumerate(E[:,word2idx[y[i]]]):
                    if prob> .1:
                        new_has_values.add((prob_i))
                        possible.add(prob_i)
            else:
                for prob_i,prob in enumerate(unknown_words(tag2idx,y[i])):
                    if prob> .01:
                        new_has_values.add((prob_i))
                        possible.add(prob_i)
        if debug:
            print(y[i],[(idx2tag[valu[0]],idx2tag[valu[1]]) for valu in has_values], [idx2tag[valu] for valu in possible])
            # if len(possible) == 0:
            #     print(emissions)
        has_values = new_has_values
    ret = np.zeros(length_of_sentence)
    ret[-1] = tag2idx["<STOP>"]
    curr = np.argmax(v[:,tag2idx["<STOP>"],length_of_sentence-1])
    for i in reversed(range(1,length_of_sentence)):
        ret[i-1] = int(curr)
        curr = int(bp[0,curr,i])
    return ret
    

    
def viterbi4(y, T, E, tag2idx, word2idx, idx2tag):
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
    debug = 0
    log_transition = np.log(T)
    log_emission = np.log(E)
    possible_states = T.shape[0]
    N = T.shape[0]
    length_of_sentence = len(y)
    v = np.zeros((possible_states,possible_states,possible_states,length_of_sentence))
    v += float('-inf')
    bp = np.zeros((possible_states,possible_states,possible_states, length_of_sentence))
    v[tag2idx["O"],tag2idx["O"], tag2idx["O"],0] = 1
    has_values = [(tag2idx["O"], tag2idx["O"])]


    for i in range(1, length_of_sentence):
        new_has_values = set()
        possible = set()
        for value2, value3 in has_values:
            if y[i] not in word2idx:
                emissions = np.log(np.array(unknown_words(tag2idx, y[i])))
                emissions = emissions[np.newaxis,:]
            else:
                emissions = log_emission[np.newaxis,:,word2idx[y[i]]]
            transition = log_transition[:,value2,value3,:]
            prev = v[:,value2,value3, i-1] 
            v[value2,value3,:,i] = np.max(prev + transition.T + emissions.T, 1)
            bp[:,value2,value3,i] = np.argmax(prev + transition.T, 1)

            if y[i] in word2idx:
                for prob_i,prob in enumerate(E[:,word2idx[y[i]]]):
                    if prob> .1:
                        new_has_values.add((value3, prob_i))
                        possible.add(prob_i)
            else:
                for prob_i,prob in enumerate(unknown_words(tag2idx,y[i])):
                    if prob> .01:
                        new_has_values.add((value3, prob_i))
                        possible.add(prob_i)
        if debug:
            print(y[i],[(idx2tag[valu[0]],idx2tag[valu[1]]) for valu in has_values], [idx2tag[valu] for valu in possible])
            # if len(possible) == 0:
            #     print(emissions)
        has_values = new_has_values
    ret = np.zeros(length_of_sentence)
    ret[-1] = tag2idx["<STOP>"]
    values = np.nan_to_num(v[:,:,tag2idx["<STOP>"],length_of_sentence-1], neginf = -1000000000 ) 

    prev = np.argmax(sum(values))
    prev2 = np.argmax(values[:,prev])
    if debug:
        print(tag2idx)
    
    #print(v[:,tag2idx["."], tag2idx["<STOP>"], length_of_sentence-1])
    # print(values[:,tag2idx["."]])
    for i in reversed(range(1,length_of_sentence)):
        ret[i-1] = int(prev)
        temp = prev2
        prev2 = int(bp[0,prev2,prev,i])
        prev = temp
    print("hey", sum(ret))
    return ret

def linear_interpolation(unigram_c, bigram_c, trigram_c):
    lambda1 = 0
    lambda2 = 0
    lambda3 = 0
        
    for a in range(len(trigram_c)):
        for b in range(len(trigram_c)):
            for c in range(len(trigram_c)):
                v = trigram_c[(a, b, c)]
                if v > 0:
                    try:
                        c1 = float( v-1 ) / ( bigram_c[(a, b)]-1 )
                    except ZeroDivisionError:
                        c1 = 0
                    try:
                        c2 = float( bigram_c[(a, b)]-1 ) / ( unigram_c.sum(axis = 1)[(a,)]-1 ) 
                    except ZeroDivisionError:
                        c2 = 0
                    try:
                        c3 = float( unigram_c.sum(axis = 1)[(a,)]-1 ) / unigram_c.sum(axis = 1).sum(axis = 0) - 1 
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
    return [norm_w, weights]

              
def unknown_words( tag2idx, word ):
    # Create empty matrix of shape (all_tags, :)
    empty_matrix = np.zeros( len( tag2idx),  ) 
    # If word is a digit, then assign 1 probability of it being `CD`
    if re.search(r'\d', word):
        empty_matrix[ tag2idx['CD'], ] = 1
    # if word has ending that looks like noun
    elif re.search(r'(ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)', word):
        empty_matrix[ tag2idx['NN'],   ] = 0.25
        empty_matrix[ tag2idx['NNS'],  ] = 0.25
        empty_matrix[ tag2idx['NNP'],  ] = 0.25
        empty_matrix[ tag2idx['NNPS'], ] = 0.25
    # If looks like gerund verb
    elif re.search(r'(ing\b)', word):
        empty_matrix[ tag2idx['VBG'], ] = 1
    # if word looks like a verb
    elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem)', word):
        empty_matrix[ tag2idx['VB'],  ] = 0.36
        empty_matrix[ tag2idx['VBD'], ] = 0.36
        empty_matrix[ tag2idx['VBN'], ] = 0.1
        empty_matrix[ tag2idx['VBP'], ] = 0.12
        empty_matrix[ tag2idx['VBZ'], ] = 0.06
    # if word's ending looks like an adjective
    elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)', word):
        empty_matrix[ tag2idx['JJ'], ] = 1
    else:
        empty_matrix[ tag2idx['NN'],  ] = 0.4
        empty_matrix[ tag2idx['VB'], ] = 0.3
        empty_matrix[ tag2idx['JJ'], ] = 0.3

    return empty_matrix
        






