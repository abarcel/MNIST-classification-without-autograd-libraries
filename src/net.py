#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import jax.numpy as np
import numpy as np

def predict(params, x):
    out_pred = [x]
    for w, b in params[:-1]:
        x_op = np.dot(x, w) + b 
        x = np.tanh(x_op)
        out_pred.extend([x])
        
    wf, bf = params[-1]
    xf = np.dot(x, wf) + bf
    return 1 / (1 + np.exp(-xf)), out_pred

def loss(params, x, y):
    y_pred = predict(params, x)[0]
    return -np.mean(np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred), axis=1))

def accuracy(params, x, y):
    y_max = np.argmax(y, axis=1)
    y_pred_max = np.argmax(predict(params, x)[0], axis=1)
    return np.mean(y_max == y_pred_max)

