# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 01:18:04 2020
Main script for program.


@author: mlazo
"""
# Library imports
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import matplotlib.pyplot as plt

tf.get_logger().setLevel('INFO')

# Local imports
from constants import *
import stf1 as stf
import SpaceBox as sb
import r2plot as r2

# Network parameters
# generation
train       = 200000    # size of synthetic training set
val         = 512       # size of synthetic validation set
window      = 128       # size of data window, number of points in one data set; 128 is default
# training
BATCH       = 256       # size of training batches
BUFFER      = 10000     # size of shuffling buffer

EPOCHS      = 3       # number of epochs to train over
EVAL_INT    = 200       # keras evaluation interval
# predictions
num         = 5000      # number of predictions to make

def normalize(a):
    '''
    Normalize an array of values to [0, 1].
    Used to prepare data for use with Tensorflow.

    Parameters
    ----------
    a : array-like
        Array of values to be normalized.

    Returns
    -------
    norm : double array
        Normalized array.
    a : double array
        Input array, unnormalized.

    '''
    ra = np.nanmax(a) - np.nanmin(a)
    norm = (a[:] - np.nanmin(a)) / ra
    
    return norm

print("Select an STF-1 data file to use:\n",
          "0: January 16, 2019\n",
          "1: February 27, 2019\n",
          "2: April 5, 2019\n",
          "3: April 19, 2019\n")
files = [0, 1, 2, 3]
try:
    file = int(input())
    if not file in files:
        raise NameError
except:
    print("Input error or file note found. Defaulting to January 19, 2019...")
    file = 0

cdsec, real_vol, real_cur, real_traces = stf.read_data(file)
window = real_vol.shape[1]

r_labels = np.zeros((real_traces, 2))
print('Retrieving magnetospheric parameters...')
r_labels[:, 0], r_labels[:, 1], real_ni, real_Vf, beta_m, beta_b = stf.analyze_data(real_vol, real_cur, real_traces)

real_vol = normalize(real_vol)
real_cur = normalize(real_cur)
real_Te_norm = normalize(r_labels[:,0])
real_Is_norm = normalize(r_labels[:,1])

print('\nReal data shapes:\nTimestamps: {}\nVoltage: {}\nCurrent: {}\n'.format(cdsec.shape, real_vol.shape, real_cur.shape))
print('Traces: {}\nData window: {}\n'.format(real_traces, window))
# print('Real labels: {}\nNormalized labels:\n{}\n{}'.format(r_labels, real_Te_norm, real_Is_norm))
print('STF-1 data imported successfully.\nBuilding real data tensors...\n')

T_r = tf.convert_to_tensor(real_Te_norm[:real_traces, None])
I_r = tf.convert_to_tensor(real_Is_norm[:real_traces, None])
label_r = tf.concat([T_r, I_r], 1)
voltage_r = tf.convert_to_tensor(real_vol[:real_traces, None])
current_r = tf.convert_to_tensor(real_cur[:real_traces, None])
data_r = tf.concat([voltage_r, current_r], 1)
print('Real tensor shapes:\nLabels: {}\nData: {}\n'.format(label_r.shape, data_r.shape))

real_set = tf.data.Dataset.from_tensor_slices((data_r, label_r))



print('Generating synthetic data...\n')
# Create synthetic data with SpaceBox model
t_labels = np.zeros((train + val, 2))
time = np.zeros((train + val, window))
vol = np.zeros((train + val, window))
cur = np.zeros((train + val, window))

period = 3 # PLACEHOLDER
box = sb.SpaceBox(T_min, T_max, Is_min, Is_max, mu_o, A_P, window, period, (train+val))
time, vol, cur, t_labels[:,0], t_labels[:,1] = box()

vol = normalize(vol)
cur = normalize(cur)
Te_norm = normalize(t_labels[:,0])
Is_norm = normalize(t_labels[:,1])

print('Training data shapes:\nTimes: {}\nVoltage: {}\nCurrent: {}\n'.format(time.shape, vol.shape, cur.shape))
print('Synthetic traces:', (train+val))
# print('Training labels: \n{}\nNormalized labels: \n{}\n{}'.format(t_labels, Te_norm, Is_norm))
print('Synthetic data successfully generated.\nBuilding synthetic data tensors...\n')

# Build training tensors
T_t = tf.convert_to_tensor(Te_norm[:train, None])
I_t = tf.convert_to_tensor(Is_norm[:train, None])
label_t = tf.concat([T_t, I_t], 1)
voltage_t = tf.convert_to_tensor(vol[:train, None])
current_t = tf.convert_to_tensor(cur[:train, None])
data_t = tf.concat([voltage_t, current_t], 1)
# Build validation tensors
T_v = tf.convert_to_tensor(Te_norm[train:, None])
I_v = tf.convert_to_tensor(Is_norm[train:, None])
label_v = tf.concat([T_v, I_v], 1)
voltage_v = tf.convert_to_tensor(vol[train:, None])
current_v = tf.convert_to_tensor(cur[train:, None])
data_v = tf.concat([voltage_v, current_v], 1)
print('Training tensor shapes:\nLabels: {}\nData: {}\n'.format(label_t.shape, data_t.shape))
print('Validation tensor shapes:\nLabels: {}\nData: {}\n'.format(label_v.shape, data_v.shape))

train_set = tf.data.Dataset.from_tensor_slices((data_t, label_t))
train_set = train_set.cache().shuffle(BUFFER).batch(BATCH).repeat()

val_set = tf.data.Dataset.from_tensor_slices((data_v, label_v))
val_set = val_set.cache().shuffle(BUFFER).batch(BATCH).repeat()

mse = k.losses.MeanSquaredError()
try:
    print('Loading saved model...')
    reconstructed_model = k.models.load_model('model')
    model = reconstructed_model
    print(model.summary())
    
    # Predictions on validation set
    xval = data_v[:num]
    yval = label_v[:num, :] # labels
    yval_p = model.predict(xval[:num]) # network predicted values, UNTRAINED
    print('Validation data shape: {}\nValidation label shape: {}\nValidation prediction shape: {}\n'.format(xval.shape, yval.shape, yval_p.shape))
    
    # Train the network!
    print('Training the network...\n')
    history = model.fit(train_set, epochs=EPOCHS,
                        steps_per_epoch=EVAL_INT,
                        validation_data=val_set, validation_steps=50,
                        callbacks=[tfdocs.modeling.EpochDots()])
    model.save('model')
    
    # Plot loss
    print('Constructing loss plot over {} epochs...'.format(EPOCHS))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss)+1)
    plt.figure()
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    plt.plot(epochs, val_loss, 'c-', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.savefig('loss_plot.png', format='png')
    print('Loss plot saved as "loss_plot.png".\nMaking predictions...\n')
    
    yval_pp = model.predict(xval[:num]) # network predicted value, TRAINED
    
    # print('Untrained prediction tensor shapes:', xval.shape, yval.shape, yval_p.shape)
    # print('Trained prediction tensor shapes: //, //,', yval_pp.shape)
    
    # Plot results with R squared
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yval[:,0], yval_p[:,0], 'T_e',
                  title=r'Network Performance on Synthetic T_e', res=40)
    
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yval[:,1], yval_p[:,1], 'I_s',
                  title=r'Network Performance on Synthetic I_s', res=40)
    
    xreal = data_r[:num]
    yreal = label_r[:num]
    yreal_p = model.predict(xreal[:num]) # Predictions on NSTX shot 137622
    
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yreal[:,0], yreal_p[:,0], 'T_e',
                  title=r'Network Performance on Real T_e', res=40)
    
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yreal[:,1], yreal_p[:,1], 'I_s',
                  title=r'Network Performance on Real I_s', res=40)
except IOError:
    print("No saved model found. Creating a model...")
    model = k.models.Sequential([
        layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=data_t.shape[-2:])),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dense(16, activation='sigmoid'),
        layers.Dense(4, activation='tanh'),
        layers.Dense(2, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    print(model.summary())
    
    # Predictions on validation set
    xval = data_v[:num]
    yval = label_v[:num, :] # labels
    yval_p = model.predict(xval[:num]) # network predicted values, UNTRAINED
    print('Validation data shape: {}\nValidation label shape: {}\nValidation prediction shape: {}\n'.format(xval.shape, yval.shape, yval_p.shape))
    
    # Train the network!
    history = model.fit(train_set, epochs=EPOCHS,
                        steps_per_epoch=EVAL_INT,
                        validation_data=val_set, validation_steps=50,
                        callbacks=[tfdocs.modeling.EpochDots()])
    model.save('model')
    
    # Plot loss
    print('Constructing loss plot over {} epochs...'.format(EPOCHS))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss)+1)
    plt.plot(epochs, train_loss, 'r-', label='Training Loss')
    plt.plot(epochs, val_loss, 'c-', label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.savefig('loss_plot.png', format='png')
    print('Loss plot saved as "loss_plot.png".\nMaking predictions...\n')
    
    yval_pp = model.predict(xval[:num]) # network predicted value, TRAINED
    
    # print('untrained prediction tensor shapes:', xval.shape, yval.shape, yval_p.shape)
    # print('trained prediction tensor shapes: //, //,', yval_pp.shape)
    
    # Plot results with R squared
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yval[:,0], yval_p[:,0], 'T_e',
                  title=r'Network Performance on Synthetic T_e', res=40)
    
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yval[:,1], yval_p[:,1], 'I_s',
                  title=r'Network Performance on Synthetic I_s', res=40)
    
    xreal = data_r[:num]
    yreal = label_r[:num]
    yreal_p = model.predict(xreal[:num]) # Predictions on STF-1 data
    
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yreal[:,0], yreal_p[:,0], 'T_e',
                  title=r'Network Performance on Real r"T_e"', res=40)
    
    fig, ax = plt.subplots()
    r2.plotResult(fig, ax, yreal[:,1], yreal_p[:,1], 'I_s',
                  title=r'Network Performance on Real I_s', res=40)
except ImportError:
    print("Save file not found. Please try again.")

plt.show()