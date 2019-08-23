import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
import dash_core_components as core
import dash_html_components as html
import sys
import pickle
from keras.models import model_from_json
from keras import backend as K
from models import Models

MAX_SEQ_LEN = 100
UNIT_DIM = 64

json_file = open('Models/sampling_model.json', 'r')
sampling_model_json = json_file.read()
json_file.close()
sampling_model = model_from_json(sampling_model_json)
sampling_model.load_weights('Models/sampling_model.h5')

word2idx, idx2word = {}, {}

with open('Models/word2idx.pickle', 'rb') as f:
    word2idx = pickle.load(f)

with open('Models/idx2word.pickle', 'rb') as f:
    idx2word = pickle.load(f)

for each in range(20):
    output_sentence = Models.sampleFromModel(sampling_model, word2idx, idx2word, MAX_SEQ_LEN, UNIT_DIM)
    print(output_sentence)
