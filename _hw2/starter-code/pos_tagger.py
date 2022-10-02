from multiprocessing import Pool
import numpy as np
import time
from utils import *
np.set_printoptions(suppress=True) # disable scientific notation
import string


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
    processes = 10
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    # unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:0 for i in range(n)}
         
    start = time.time()
    ans = []
    for i in range(0,n):
        predictions[i] = model.inference(sentences[i])
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    
    predictions = dict()
    
    for a in ans:
        predictions.update(a)
    
    print(f"Inference Runtime: {round((time.time()-start)/60, 3)} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        # print(i)
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {round( (time.time()-start)/60, 3)} minutes.")


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
    print("Whole sent acc: {}".format(round(whole_sent_acc/num_whole_sent, 3)))
    print("Mean Probabilities: {}".format(round( sum(probabilities.values())/n, 3)) )
    print("Token acc: {}".format(round( token_acc, 3)))
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
        self.unigrams = self.unigrams / self.unigrams.sum(axis = 1).sum(axis = 0)

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
    def preprocessing(self,data):
        sentence_data = data[0]
        tag_data = data[1]
        for i in range(len(sentence_data)):
            sentence = sentence_data[i]
            sentence_tags = tag_data[i]
            
            new_sentence = []
            new_tags = []
            for n,word in enumerate(sentence):
                if word not in string.punctuation:
                    new_sentence.append(word)
                    new_tags.append(sentence_tags[n])
            sentence_data[i] = new_sentence
            tag_data[i] = new_tags
        return [sentence_data,tag_data]

    
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
        self.sequence = sequence    
        
        ## TODO
        return 0. 
        pass

    def is_punctuation(self,word):
        return not word in string.punctuation

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.
        You should implement different kinds of inference (suggested as separate
        methods):
            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        # GREEDY = 0; BEAM_1 = 1; BEAM_2 = 2; BEAM_3 = 3; VITERBI2 = 4; VITERBI3 = 5
        flag = VITERBI2

        # GREEDY
        if flag == GREEDY:
            idxseq = []
            unknown = []
            for word in sequence:
                if (word in self.all_words) == True:
                    idxseq.append(self.lexical[:,self.word2idx[word]])
                    # data = np.array(idxseq).T
                else:
                    idxseq.append(np.array(unknown_words( self.tag2idx, word)) )  
                    # data = np.array(idxseq).T
            data = np.array(idxseq).T
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

        # Beam Search
        if flag == BEAM_1:
                i = []
                for word in sequence:
                    i.append( self.word2idx[ word ] )
                data = self.lexical[:,i]
                data = np.array(data).T
                # print( data.T )
                sequences = [[list(), 0.0]]
                # walk over each step in sequence
                for row in data:
                    all_candidates = list()
                    # expand each current candidate
                    for i in range(len(sequences)):
                        seq, score = sequences[i]
                        for j in range(len(row)):
                            candidate = [seq + [j], score - log(row[j])]
                            all_candidates.append(candidate)
                        # order all candidates by score
                        ordered = sorted(all_candidates, key=lambda tup:tup[1])
                            # select k best
                        sequences = ordered[:1]
                        pred_tags = []
                        for i in sequences[0][0]:
                            pred_tags.append(self.idx2tag[i])
                # return( [ sequences[0][0], sequences[1][0], sequences[2][0] ] )
                return( pred_tags )
            
        # Beam Search 2
        if flag == BEAM_2:
                i = []
                for word in sequence:
                    i.append( self.word2idx[ word ] )
                data = self.lexical[:,i]
                data = np.array(data).T
                # print( data.T )
                sequences = [[list(), 0.0]]
                # walk over each step in sequence
                for row in data:
                    all_candidates = list()
                    # expand each current candidate
                    for i in range(len(sequences)):
                        seq, score = sequences[i]
                        for j in range(len(row)):
                            candidate = [seq + [j], score - log(row[j])]
                            all_candidates.append(candidate)
                        # order all candidates by score
                        ordered = sorted(all_candidates, key=lambda tup:tup[1])
                            # select k best
                        sequences = ordered[:2]
                        pred_tags = []
                        for i in sequences[1][0]:
                            pred_tags.append(self.idx2tag[i])
                # return( [ sequences[0][0], sequences[1][0], sequences[2][0] ] )
                return( pred_tags )

        # Beam Search 2
        if flag == BEAM_3:
                i = []
                for word in sequence:
                    i.append( self.word2idx[ word ] )
                data = self.lexical[:,i]
                data = np.array(data).T
                # print( data.T )
                sequences = [[list(), 0.0]]
                # walk over each step in sequence
                for row in data:
                    all_candidates = list()
                    # expand each current candidate
                    for i in range(len(sequences)):
                        seq, score = sequences[i]
                        for j in range(len(row)):
                            candidate = [seq + [j], score - log(row[j])]
                            all_candidates.append(candidate)
                        # order all candidates by score
                        ordered = sorted(all_candidates, key=lambda tup:tup[1])
                            # select k best
                        sequences = ordered[:3]
                        pred_tags = []
                        for i in sequences[1][0]:
                            pred_tags.append(self.idx2tag[i])
                # return( [ sequences[0][0], sequences[1][0], sequences[2][0] ] )
                return( pred_tags )

        # VITERBI
        ret = []
        if flag == VITERBI2 or flag == VITERBI3:
            idxseq = []
            for word in sequence:
                idxseq.append(self.word2idx[word])
            # Bigram Viterbi
            if flag == VITERBI2:
                x = viterbi2(idxseq, self.bigrams, self.lexical, self.tag2idx)
            # Trigram Viterbi
            elif flag == VITERBI3:
                x = viterbi3(idxseq, self.trigrams, self.lexical, self.tag2idx, self.idx2word, self.idx2tag)

            for tag in x:
                ret.append(self.idx2tag[tag])
            return ret



if __name__ == "__main__":

    pos_tagger = POSTagger()



    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    # test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Printing/Testing ----------------------------------------------
    pos_tagger.get_emissions(); pos_tagger.get_bigrams(); 
    pos_tagger.get_trigrams(); pos_tagger.get_unigrams()
    
    print(pos_tagger.inference(["-docstart-","house",".","<STOP>"]))


    # from sklearn.metrics import precision_recall_fscore_support as score
    # predicted = [pos_tagger.inference( train_data[0][i]) for i in range(len(train_data[0])) ]
    # actual = train_data[1]
    # predicted = [item for sublist in predicted for item in sublist]
    # actual = [item for sublist in actual for item in sublist]
    # precision, recall, fscore, support = score(actual, predicted)
    # import statistics
    # print("The average Precision across all individual tags is", round( statistics.mean(precision), 3) )
    # print("The average Recall is ", round( statistics.mean(recall), 3))
    # print("And the average fscore is ", round( statistics.mean(fscore), 3))

    print("")
    evaluate( 
        [train_data[0][0:10], train_data[1][0:10]],
        # train_data,
        pos_tagger
     )


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



