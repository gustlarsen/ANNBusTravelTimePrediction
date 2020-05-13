import matplotlib.pyplot as plt
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
# import pickle

layer_width = [50]

n_inputs = 111
n_outputs = 36

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

dataset = pd.read_csv('Dataset.csv', decimal=',', sep=';')

dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_stats.to_csv('train_stats.csv')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

train_data_as_numpy = normed_train_data.values
test_data_as_numpy = normed_test_data.values
x_train = train_data_as_numpy[:, 0:n_inputs]
y_train = train_data_as_numpy[:, n_inputs:]
x_test = test_data_as_numpy[:, 0:n_inputs]
y_test = test_data_as_numpy[:, n_inputs:]

for width in layer_width:
    model = Sequential()
    model.add(Dense(units=width, activation='sigmoid', input_dim=n_inputs))
    model.add(Dense(units=width, activation='sigmoid'))
    model.add(Dense(units=n_outputs, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='Nadam', metrics=['mean_squared_error'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2)

    history = model.fit(x_train, y_train, validation_split=0.2, epochs=100,
                         callbacks=[early_stop], verbose=1)


    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label='Val Error')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()


    plot_history(history)

    # hist = pd.DataFrame(history.history)

    # with open('history.txt', 'wb') as file:
    #     pickle.dump(history.history, file)

    loss, mse = model.evaluate(x_test, y_test, verbose=0)

    print('loss ', loss, 'mse', mse)

    outputs = model.predict(x_test, verbose=0)

    correlations = np.zeros(outputs.shape[1])

    for i in range(len(correlations)):
        correlations[i] = np.corrcoef(y_test[:, i], outputs[:, i])[0, 1]

    # print(normed_test_data.columns.values[26:])
    print(correlations)

    model.save('calibration_nn.h5')
    # del model
