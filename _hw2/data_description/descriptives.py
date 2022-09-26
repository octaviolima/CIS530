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
import os, sys
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


#----------------------------------------------------------------
# Calculate Number of characters - statistcs
#----------------------------------------------------------------



























































