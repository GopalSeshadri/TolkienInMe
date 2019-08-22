import numpy as np
import pandas as pd
import keras
import sys
import re
from preprocess import Preprocess

MAX_VOCAB_SIZE = 5000
MAX_SEQ_LEN = 100
EMBEDDING_DIM = 100

input_data, output_data, full_data = [], [], []
input_data.extend(Preprocess.readData('robert_frost.txt')[0])
output_data.extend(Preprocess.readData('robert_frost.txt')[1])
full_data.extend(Preprocess.readData('robert_frost.txt')[2])

print(full_data[0])

tokenizer, word2idx = Preprocess.fitTokenizer(full_data, MAX_VOCAB_SIZE)
input_seq = tokenizer.texts_to_sequences(input_data)
output_seq = tokenizer.texts_to_sequences(output_data)

print(input_seq[0])

max_seq_len = max([len(each) for each in input_seq])
max_seq_len = min(max_seq_len, MAX_SEQ_LEN)
input_seq = Preprocess.padSequences(input_seq, max_seq_len)
output_seq = Preprocess.padSequences(output_seq, max_seq_len)

print(max_seq_len)

word2vec = Preprocess.getWord2Vec(EMBEDDING_DIM)
embedding_matrix, num_words = Preprocess.getEmbeddingMatrix(MAX_VOCAB_SIZE, word2idx, word2vec)

onehot_output_seq = Preprocess.oneHotOutput(output_seq, max_seq_len, num_words)
