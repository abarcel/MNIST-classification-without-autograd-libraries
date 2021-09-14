#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from jax import jit
#import jax.numpy as np

import time
import pandas as pd
import numpy as np
from numpy import random

from src.initial_params import initial_params
from src.one_hot import one_hot
from src.net import *
from src.backpropagation import backprop

layer_sizes = [784, 1024, 1024, 10]
param_scale = 0.1
step_size = 0.05
num_epochs = 10
batch_size = 128

train = pd.read_csv('digit_recognizer/train.csv')
x_test = pd.read_csv('digit_recognizer/test.csv')
x_train = train.drop(['label'], axis = 1)
y_train = train['label'] 
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test, y_train = x_train.values, x_test.values, y_train.values
y_train = one_hot(y_train, 10)
num_train = x_train.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

def data_stream(rnd = random.RandomState(99)):
    while True:
        perm = rnd.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield batch_idx
            
batches = data_stream()

#@jit
def update(params, x, y):
    grads = backprop(params, x, y)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]


params = initial_params(param_scale, layer_sizes)
params_test = params.copy()
pre_acc = accuracy(params, x_train, y_train)
for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
        idx = next(batches)
        params = update(params, x_train[idx], y_train[idx])
    epoch_time = time.time() - start_time

train_acc = accuracy(params, x_train, y_train)
print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
print("Pre-Update Training set accuracy {}".format(pre_acc))
print("Training set accuracy {}".format(train_acc))

prediction = np.argmax(predict(params, x_test)[0], axis=1)
submission = pd.DataFrame({'ImageId' : range(1,28001), 'Label' : list(prediction)})

submission.to_csv("submission.csv",index=False)

print('"submission.csv" file created.')

