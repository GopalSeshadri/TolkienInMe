import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class Preprocess:

    def readData(filename):
        input_data = []
        output_data = []
        full_data = []
        for each in open('Data/{}'.format(filename)):
            each = each.lower().strip()
            if not each:
                continue

            input_data.append('<sos> ' + each)
            output_data.append(each + ' <eos>')
            full_data.append('<sos> ' + each + ' <eos>')

        return input_data, output_data, full_data

    def fitTokenizer(data, vocab_size):
        tokenizer = Tokenizer(num_words = vocab_size, filters = '')
        tokenizer.fit_on_texts(data)
        return tokenizer, tokenizer.word_index

    def padSequences(data, max_seq_len):
        return pad_sequences(data, maxlen = max_seq_len, padding = 'post')

    def getWord2Vec(embedding_dim):
        '''
        This function returns the dictionary of word vectors

        Returns:
        word2vec (dict) : A dictionary of word vectors
        '''
        word2vec = {}
        with open('Embeddings/glove.6B.{}d.txt'.format(embedding_dim), encoding = 'utf8') as file:
            for line in file:
                values = line.split()
                word2vec[values[0]] = np.asarray(values[1:], dtype = 'float32')
        return word2vec

    def getEmbeddingMatrix(max_vocab, word2idx, word2vec):
        '''
        This function takes in maximum vocabulary size, word2idx and word2vec and it returns the embedding matrix.

        Parameters:
        max_vocab (int) : Maximum vocabulary size.
        word2idx (dict) : A dictionary of tokenized data
        word2vec (dict) : A dictionay of word vectors

        Returns:
        embedding_matrix (numpy array) : A matrix of embedding vectors
        '''
        number_of_words = min(max_vocab, len(word2idx) + 1)
        embedding_matrix = np.zeros((number_of_words, 100)) # Here 100 is the dimension of GloVe Embeddings
        for word, idx in word2idx.items():
            if idx < max_vocab:
                embedding_vector = word2vec.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
        return embedding_matrix, number_of_words

    def oneHotOutput(output_seq, max_seq_len, num_words):
        onehot_output_seq = np.zeros((len(output_seq), max_seq_len, num_words))
        for i, each in enumerate(output_seq):
            for w, word_idx in enumerate(each):
                if word_idx > 0:
                    onehot_output_seq[i, w, word_idx] = 1
        return onehot_output_seq
