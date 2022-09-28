#####################################
# Goal: Create Data Descriptives
#####################################

#----------------------------------------------------------------
# Import Libraries + helper files
#----------------------------------------------------------------
import pandas as pd
from tqdm import tqdm 
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os, sys, re
sys.path.insert(0, 'starter-code/')
import utils, constants, statistics

#----------------------------------------------------------------
# Load in Raw Datasets as is
#----------------------------------------------------------------
train_x = pd.read_csv( "data/train_x.csv" )
train_y = pd.read_csv( "data/train_y.csv" )
dev_x   = pd.read_csv( "data/dev_x.csv"   )
dev_y   = pd.read_csv( "data/dev_y.csv"   )
test_x  = pd.read_csv( "data/test_x.csv"  )

#----------------------------------------------------------------
# Load in Raw Datasets with helper functions
#----------------------------------------------------------------
train_data = load_data( "data/train_x.csv", "data/train_y.csv" )
dev_data   = load_data( "data/dev_x.csv",   "data/dev_y.csv"   )
test_data  = load_data( "data/test_x.csv")

#----------------------------------------------------------------
# Calculate length of sentences - statistcs
#----------------------------------------------------------------
# TRAIN ------------
# Mean length of sentence
results_t = []
for i in range( len( train_data[0] ) ):
  results_t.append( len( train_data[0][i] ) )
# take mean
statistics.mean(results_t)
# take median
statistics.median(results_t)

# DEV ---------------
# Mean length of sentence
results_d = []
for i in range( len( dev_data[0] ) ):
  results_d.append( len( dev_data[0][i] ) )
# take mean
statistics.mean(results_d)
# take median
statistics.median(results_d)

# TEST ---------------
results_test = []
for i in range( len( test_data ) ):
  results_test.append( len( test_data[i] ) )
# Take mean
statistics.mean(results_test)
# Take median
statistics.median(results_test)


#----------------------------------------------------------------
# Calculate Number of characters - statistcs
#----------------------------------------------------------------
# TRAIN --------------
results_t = []
for i in range(len(train_data[0])):
  results_t.append( sum( len(word) for word in train_data[0][i] ) )
# Take mean
statistics.mean(results_t)
# take median
statistics.median(results_t)

# DEV ----------------
results_d = []
for i in range(len(dev_data[0])):
  results_d.append( sum( len(word) for word in dev_data[0][i] ) )
# Take mean
statistics.mean(results_d)
# take median
statistics.median(results_d)

# TEST ---------------
results_test = []
for i in range(len(test_data)):
  results_test.append( sum( len(word) for word in test_data[i] ) )
# Take mean
statistics.mean(results_test)
# take median
statistics.median(results_test)

#----------------------------------------------------------------
# Vocab. Size
#----------------------------------------------------------------
train_words = pd.read_csv("data/train_x.csv")
dev_words   = pd.read_csv("data/dev_x.csv")
test_words  = pd.read_csv("data/test_x.csv")

# Unique-word count ---------------
# No. of unique words from training
np.unique( np.array( train_words.word.str.lower() ) ).shape
# No. of unique words from dev
np.unique( np.array( dev_words.word.str.lower() ) ).shape
# No. of unique words from test
np.unique( np.array( test_words.word.str.lower() ) ).shape


################################################################################
#################### Lowercase & without punctuations  #########################
################################################################################

#----------------------------------------------------------------
# Calculate length of sentences - statistcs
#----------------------------------------------------------------

# TRAIN ------------
# Mean length of sentence
results_t = []
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

for i in range( len( train_data[0] ) ):
  train_data[0][i] = [''.join(c for c in s if c not in punctuation) for s in train_data[0][i]]
  train_data[0][i] = [s for s in train_data[0][i] if s]
  results_t.append( len( train_data[0][i] ) )
# take mean
statistics.mean(results_t)
# take median
statistics.median(results_t)

# DEV ------------
# Mean length of sentence
results_d = []
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

for i in range( len( dev_data[0] ) ):
  dev_data[0][i] = [''.join(c for c in s if c not in punctuation) for s in dev_data[0][i]]
  dev_data[0][i] = [s for s in dev_data[0][i] if s]
  results_d.append( len( dev_data[0][i] ) )
# take mean
statistics.mean(results_d)
# take median
statistics.median(results_d)

# TEST ------------
# Mean length of sentence
results_test = []
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

for i in range( len( test_data ) ):
  test_data[i] = [''.join(c for c in s if c not in punctuation) for s in test_data[i]]
  test_data[i] = [s for s in test_data[i] if s]
  results_test.append( len( test_data[i] ) )
# take mean
statistics.mean(results_test)
# take median
statistics.median(results_test)

#----------------------------------------------------------------
# Calculate Number of characters - statistcs
#----------------------------------------------------------------

# TRAIN --------------
results_t = []
for i in range(len(train_data[0])):
  train_data[0][i] = [''.join(c for c in s if c not in punctuation) for s in train_data[0][i]]
  train_data[0][i] = [s for s in train_data[0][i] if s]
  results_t.append( sum( len(word) for word in train_data[0][i] ) )
# Take mean
statistics.mean(results_t)
# take median
statistics.median(results_t)

# DEV ----------------
results_d = []
for i in range(len(dev_data[0])):
  dev_data[0][i] = [''.join(c for c in s if c not in punctuation) for s in dev_data[0][i]]
  dev_data[0][i] = [s for s in dev_data[0][i] if s]
  results_d.append( sum( len(word) for word in dev_data[0][i] ) )
# Take mean
statistics.mean(results_d)
# take median
statistics.median(results_d)

# TEST ---------------
results_test = []
for i in range(len(test_data)):
  test_data[i] = [''.join(c for c in s if c not in punctuation) for s in test_data[i]]
  test_data[i] = [s for s in test_data[i] if s]
  results_test.append( sum( len(word) for word in test_data[i] ) )
# Take mean
statistics.mean(results_test)
# take median
statistics.median(results_test)














































