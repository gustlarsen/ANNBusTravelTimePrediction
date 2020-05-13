from keras.models import load_model
import keras
import pandas as pd
import numpy as np

new_inputs = pd.read_csv('newdata.csv', sep=';', decimal=',')

model = load_model('calibration_nn.h5')

train_stats = pd.read_csv('train_stats.csv', index_col=0, sep=',', decimal='.')

input_stats = train_stats.drop(index=['y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
                                      'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
                                      'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29',
                                      'y30', 'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'yTotal'])


def norm(x):
    return (x - input_stats['mean']) / input_stats['std']


normed_new_inputs = norm(new_inputs)

x_numpy = normed_new_inputs.values

y_numpy = model.predict(x_numpy)

# new_outputs = pd.DataFrame(y_numpy, columns=['y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
#                                              'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
#                                              'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29',
#                                              'y30', 'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'yTotal'])
#
# output_stats = train_stats.loc[['y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11',
#                                 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
#                                 'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29',
#                                 'y30', 'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'yTotal']]
#
# def denorm(y):
#     return (y * output_stats['std']) + output_stats['mean']
#
# new_outputs = denorm(new_outputs)
#
# new_outputs.to_csv("new_outputs.csv", index=False, sep=';', decimal=',')