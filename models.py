import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Embedding, Input, Dense
from keras.layers import LSTM, GRU
from keras.optimizers import Adam
class Models:
    def usingLSTM(input_seq, onehot_output_seq, embedding_matrix, max_seq_len, unit_dim, num_words, batch_size):
        embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights = [embedding_matrix],
                                    trainable = False)

        input =  Input(shape = (max_seq_len, ))
        x = embedding_layer(input)

        initial_h = Input(shape = (unit_dim,))
        initial_c = Input(shape = (unit_dim,))

        lstm_layer = LSTM(unit_dim, return_sequences = True, return_state = True)
        x, _, _ = lstm_layer(x, initial_state = [initial_h, initial_c])

        dense_layer = Dense(num_words, activation = 'softmax')
        output = dense_layer(x)

        model = Model([input, initial_h, initial_c], output)
        model.compile(loss = 'categorical_crossentropy',
                    optimizer = Adam(lr = 1e-3, decay = 1e-6),
                    metrics = ['accuracy'])

        hidden = np.zeros((len(input_seq), unit_dim))
        cell = np.zeros((len(input_seq), unit_dim))
        model.fit([input_seq, hidden, cell],
                onehot_output_seq,
                epochs = 100,
                batch_size = batch_size,
                validation_split = 0.1)

        return model, embedding_layer, lstm_layer, dense_layer, initial_h, initial_c

    def samplingModel(embedding_layer, lstm_layer, dense_layer, initial_hidden, initial_cell):
        input = Input(shape = (1,))
        x = embedding_layer(input)
        x, hidden, cell = lstm_layer(x, initial_state = [initial_hidden, initial_cell])
        output = dense_layer(x)
        sampling_model = Model([input, initial_hidden, initial_cell], [output, hidden, cell])
        return sampling_model

    def sampleFromModel(sampling_model, word2idx, idx2word, max_seq_len, unit_dim):
        output_sentence = []
        sos_idx, eos_idx = word2idx['<sos>'], word2idx['<eos>']
        input = np.array([[sos_idx]])
        hidden = np.zeros((1, unit_dim))
        cell = np.zeros((1, unit_dim))

        for word in range(max_seq_len):
            output, hidden, cell = sampling_model.predict([input, hidden, cell])
            probs = output[0, 0]
            if np.argmax(probs) == 0:
                print("MAX")
            probs[0] = 0
            probs /= probs.sum()
            idx = np.random.choice(len(probs), p = probs)
            if idx == eos_idx:
                break

            output_sentence.append(idx2word.get(idx))
            input[0, 0] = idx

        return ' '.join(output_sentence)
