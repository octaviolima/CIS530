#----------------------------------------------------------------
# 1- Import Libraries + helper files
#----------------------------------------------------------------
library(pacman); pacman::p_load(data.table, tidyverse, magrittr, 
                                ggplot2, forcats)

#----------------------------------------------------------------
# 2- Calculate Word Frequency
#----------------------------------------------------------------

# Training ------

train = fread(
        "data/train_x.csv"
        )[word != "-DOCSTART-" & 
            word != "<STOP>"]

train[, word := tolower(word)]

train$word = gsub('[[:punct:] ]+', NA, train$word)

freq_t = train %>% 
        group_by(word) %>% 
        tally() %>% 
        arrange(desc(n)) %>% 
        data.table

ggplot(
  freq_t[0:50][is.na(word) == FALSE],
  aes(x = reorder(word, n), y = n)
  ) + ylab(NULL) +
  theme(axis.text.y = element_blank()) +
  geom_bar( stat = 'identity', fill = 'black') +
  coord_flip() + theme_minimal() +
  ggtitle("Word Frequency in Training")

# DEV ---------

dev = fread(
  "data/dev_x.csv"
)[word != "-DOCSTART-" & 
    word != "<STOP>"]

dev[, word := tolower(word)]

dev$word = gsub('[[:punct:] ]+', NA, dev$word)

freq_d = dev %>% 
  group_by(word) %>% 
  tally() %>% 
  arrange(desc(n)) %>% 
  data.table

ggplot(
  freq_d[0:50][is.na(word) == FALSE],
  aes(x = reorder(word, n), y = n)
) + ylab(NULL) +
  theme(axis.text.y = element_blank()) +
  geom_bar( stat = 'identity', fill = 'black') +
  coord_flip() + theme_minimal() +
  ggtitle("Word Frequency in Development")

# Combine two graphs --------

grid.arrange(
  ggplot(
    freq_t[0:50][is.na(word) == FALSE],
    aes(x = reorder(word, n), y = n)
  ) + ylab(NULL) +
    theme(axis.title.y = element_blank()) +
    geom_bar( stat = 'identity', fill = 'black') +
    coord_flip() + theme_minimal() +
    ggtitle("Word Frequency in Training"),
  
  
  ggplot(
    freq_d[0:50][is.na(word) == FALSE],
    aes(x = reorder(word, n), y = n)
  ) + ylab(NULL) +
    theme(axis.text.y = element_blank()) +
    geom_bar( stat = 'identity', fill = 'black') +
    coord_flip() + theme_minimal() +
    ggtitle("Word Frequency in Development"),
  ncol = 2, nrow = 1
  
)

#----------------------------------------------------------------
# 3- Calculate Mean/Median Length of sentences
#----------------------------------------------------------------

# TRAIN -----














