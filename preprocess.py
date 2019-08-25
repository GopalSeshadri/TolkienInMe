import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import re
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class Preprocess:

    def getFiles():
        '''
        This function returns the list of files in data folder.

        Returns:
        text_list (list) : A list of file names
        '''
        text_list = []
        for folder in os.listdir('Data'):
            for file in os.listdir('Data/' + folder):
                if file[-3:] == 'txt':
                    name = '{}/{}'.format(folder, file)
                    text_list.append(name)
        return text_list

    def readData(filename):
        '''
        This function takes in a filename and returns three lists, one for input of the model, one for the output and one for tokenization.

        Parameters:
        filename (str) : The name of the input file.

        Returns:
        input_data (list) : A list of sentences with <sos> tag at the beginning
        output_data (list) : A list of sentences with <eos> tag at the end
        full_data (list) : A list of sentences with <sos> tag and <eos> tag. This is used to train the tokenizer.
        '''
        input_data = []
        output_data = []
        full_data = []
        for each in open('Data/{}'.format(filename)).read().replace('\n', ' ').split('.'):
            each = each.lower().strip()
            each = each.translate(str.maketrans('', '', string.punctuation))
            if not each:
                continue

            input_data.append('<sos> ' + each)
            output_data.append(each + ' <eos>')
            full_data.append('<sos> ' + each + ' <eos>')

        return input_data, output_data, full_data

    def fitTokenizer(data, vocab_size):
        '''
        This function takes in data and vocabulary size and train a tokenizer and returns it and word to index of the same.

        Parameters:
        data (list) : The data for tokenizer to train on.
        vocab_size (int) : The size of the vocalbulary, which is the number of words in the tokenizer.

        Returns:
        tokenizer (Object) : A tokenizer object fitted on the above data.
        '''
        tokenizer = Tokenizer(num_words = vocab_size, filters = '')
        tokenizer.fit_on_texts(data)
        return tokenizer, tokenizer.word_index

    def padSequences(data, max_seq_len):
        '''
        This function takes in data and the maximum sequence length. Returns list with fixed sequence length.

        Parameters:
        data (list) : A list of sequences with varied lengths.
        max_seq_len (int) : The maximum length of the input sequence.
        '''
        return pad_sequences(data, maxlen = max_seq_len, padding = 'post')

    def getWord2Vec(embedding_dim):
        '''
        This function returns the dictionary of word vectors

        Parameters:
        embedding_dim (int) : The size of the embedding vectors

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
        '''
        This function takes in the output sequence list, the maximum length of sequences and the number of words. Returns one hot version
        of the given output sequence.

        Parameters:
        output_seq (list) : A list of sequence with fixed length
        max_seq_len (int) : The maximum length of the given input sequences
        num_words (int) : The number of words in the vocabulary.

        Returns:
        onehot_output_seq (int) : The list of one hot vectors for the given list of output sequence.
        '''
        onehot_output_seq = np.zeros((len(output_seq), max_seq_len, num_words))
        for i, each in enumerate(output_seq):
            for w, word_idx in enumerate(each):
                if word_idx > 0:
                    onehot_output_seq[i, w, word_idx] = 1
        return onehot_output_seq
