from word_vectors import WordVectors                                                                                
import numpy as np
import sys

# syn_final.npy: storing word embeddings, numpy array of shape [vocab_size, hidden_size]
# 'vocab.txt': text file storing words in vocabulary, one word per line

model_folder = sys.argv[1]
query = sys.argv[2]
num_similar_words = 10
syn0_final = np.load(model_folder + '/syn0_final.npy')
vocab_words = []                                                                                                   
with open(model_folder + '/vocab.txt') as f:
    vocab_words = [l.strip() for l in f] 
                                                                                                                    
wv = WordVectors(syn0_final, vocab_words)   
print(wv.most_similar(query, num_similar_words))
