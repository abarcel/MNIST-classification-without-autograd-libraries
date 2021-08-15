#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import jax.numpy as np
import numpy as np

sub_delta = lambda p, d: np.sum(p*d[:,np.newaxis,:], axis=2)
param_w = lambda o, d: np.mean(o[...,np.newaxis]*d[:,np.newaxis,:], axis=0)

def backprop(params, x, y):
    y_pred, out_pred = predict(params, x)
    new_params = []
    delta = (-(y/y_pred) + (1 - y)/(1 - y_pred)) * (y_pred - y_pred**2)
    for i in range(len(layer_sizes)-2, -1, -1):
        new_params.extend([(param_w(out_pred[i], delta), np.mean(delta, axis=0))])
        delta = (1 - out_pred[i]**2) * sub_delta(params[i][0], delta)
    return new_params[::-1]

