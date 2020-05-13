# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:34:53 2019

@author: gusta
"""

from keras.models import load_model
import keras
import pandas as pd
import numpy as np

new_inputs = pd.read_csv('newdata1.csv', sep=';', decimal=',')

model = load_model('calibration_nn.h5')

train_stats = pd.read_csv('train_stats.csv', index_col=0, sep=',', decimal='.')

input_stats = train_stats.drop(index=['y2'])


def norm(x):
    #return (x - input_stats['min']) / (input_stats['max'] - input_stats['min']) 
    return (x - input_stats['mean']) / input_stats['std']


normed_new_inputs = norm(new_inputs)

x_numpy = normed_new_inputs.values

y_numpy = model.predict(x_numpy)

new_outputs = pd.DataFrame(y_numpy, columns=['y2'])

output_stats = train_stats.loc[['y2']]


def denorm(y):
    return (y * output_stats['std']) + output_stats['mean']


new_outputs = denorm(new_outputs)

new_outputs.to_csv("new_outputs.csv", index=False, sep=';', decimal=',')
