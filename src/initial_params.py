#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy import random

def initial_params(scale, dense, rnd = random.RandomState(99)):
    return [(scale * rnd.randn(m, n), scale * rnd.randn(n))             for m, n in zip(dense[:-1], dense[1:])]

