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
import utils, constants

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
# Calculate Descriptives (R)
#----------------------------------------------------------------
library(pacman); pacman::p_load(data.table, tidyverse, magrittr, ggplot2)

# Load data and remove "start", "stop"
train = fread( "data/train_x.csv" )[ word != "-DOCSTART-" & word != "<STOP>" ]
# turn words to lower case
train[, word := tolower(word)]
# remove punctuations
train$word = gsub('[[:punct:] ]+', NA, train$word)
# get frequency
freq = train %>% group_by(word) %>% tally() %>% arrange(desc(n)) %>% data.table
# plot
ggplot(freq[0:50], aes(x = word, y = n)) + geom_bar(stat = 'identity') + coord_flip()































































