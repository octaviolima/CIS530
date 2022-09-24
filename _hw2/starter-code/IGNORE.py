from collections import defaultdict
import gzip, os, statistics
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pandas as pd
from tqdm import tqdm 
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from constants import *

# Change work directory
os.chdir("/Users/octavioelias/Documents/_fall22/_hw2/")

# Def Load Data
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

# Load data
train_data = load_data('data/train_x.csv', 'data/train_y.csv')
