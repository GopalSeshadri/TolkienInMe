import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
import dash_core_components as core
import dash_html_components as html
from dash.dependencies import Input, Output
import sys
import pickle
from keras.models import model_from_json
from keras import backend as K
from models import Models

app = dash.Dash()

def generateText(sampling_model, word2idx, idx2word, MAX_SEQ_LEN, UNIT_DIM):
    output_list = []
    for each in range(20):
        output_sentence = Models.sampleFromModel(sampling_model, word2idx, idx2word, MAX_SEQ_LEN, UNIT_DIM)
        output_sentence = output_sentence.capitalize()
        output_list.append(output_sentence)

    output_text = '. '.join(output_list)
    return output_text

app.layout = html.Div([
    html.H1(id = 'h1-text',
            children = 'Mimicking Tolkien',
            style = {'height' : '20%',
                    'fontStyle' : 'italic'}),

    html.Button(id = 'button-generate',
                children = 'Suprise Me'),

    html.P(id = 'p-text',
        children = 'Your text will be generated here ...',
        style = {'fontSize' : '24px',
                'fontStyle' : 'italic',
                'height' : '40%'}),

    html.P(id = 'p2-text',
        style = {'fontSize' : '24px',
                'fontStyle' : 'italic',
                'height' : '40%'})
], style = {'width' : '100%',
            'height' : '100%',
            'backgroundColor' : '#cc9900'})

@app.callback([Output('p-text', 'children'),
            Output('p2-text', 'children')],
            [Input('button-generate', 'n_clicks')])
def affectP(n_clicks):
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

    para1_text = generateText(sampling_model, word2idx, idx2word, MAX_SEQ_LEN, UNIT_DIM)
    para2_text = generateText(sampling_model, word2idx, idx2word, MAX_SEQ_LEN, UNIT_DIM)

    K.clear_session()

    return para1_text, para2_text


if __name__ == '__main__':
    app.run_server()
