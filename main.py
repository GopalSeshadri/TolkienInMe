import numpy as np
import pandas as pd
import keras
import sys
import re
from preprocess import Preprocess
from models import Models
import pickle

MAX_VOCAB_SIZE = 3000
MAX_SEQ_LEN = 50
EMBEDDING_DIM = 100
UNIT_DIM = 64
BATCH_SIZE = 16

## Reading from Data files
input_data, output_data, full_data = [], [], []
files = Preprocess.getFiles()
print(files)
for each in files:
    input_data.extend(Preprocess.readData(each)[0])
    output_data.extend(Preprocess.readData(each)[1])
    full_data.extend(Preprocess.readData(each)[2])

## Fitting the tokenizer and tokenizing the input and output sequnce
tokenizer, word2idx = Preprocess.fitTokenizer(full_data, MAX_VOCAB_SIZE)
idx2word = {word2idx[each] : each for each in word2idx.keys()}
input_seq = tokenizer.texts_to_sequences(input_data)
output_seq = tokenizer.texts_to_sequences(output_data)

## Padding the input and output sequences
max_seq_len = max([len(each) for each in input_seq])
max_seq_len = min(max_seq_len, MAX_SEQ_LEN)
input_seq = Preprocess.padSequences(input_seq, max_seq_len)
output_seq = Preprocess.padSequences(output_seq, max_seq_len)

# print(max_seq_len)

## Getting the word2vec Glove vectors and embedding matrix
word2vec = Preprocess.getWord2Vec(EMBEDDING_DIM)
embedding_matrix, num_words = Preprocess.getEmbeddingMatrix(MAX_VOCAB_SIZE, word2idx, word2vec)

## Convert the output sequnce to one hot vectors to use categorical_crossentropy loss function
onehot_output_seq = Preprocess.oneHotOutput(output_seq, max_seq_len, num_words)

## Fitting the models
model, embedding_layer, lstm_layer, dense_layer, hidden, cell = Models.usingLSTM(input_seq, onehot_output_seq, embedding_matrix, max_seq_len, UNIT_DIM, num_words, BATCH_SIZE)
sampling_model = Models.samplingModel(embedding_layer, lstm_layer, dense_layer, hidden, cell)

## Saving the models
model_json = sampling_model.to_json()
with open('Models/sampling_model.json', 'w') as json_file:
    json_file.write(model_json)
sampling_model.save_weights('Models/sampling_model.h5')

with open('Models/word2idx.pickle', 'wb') as f:
    pickle.dump(word2idx, f, protocol = pickle.HIGHEST_PROTOCOL)

with open('Models/idx2word.pickle', 'wb') as f:
    pickle.dump(idx2word, f, protocol = pickle.HIGHEST_PROTOCOL)
