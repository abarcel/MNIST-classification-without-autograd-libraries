#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def one_hot(x, k, dtype=np.float32):
    return np.array(x[:, None] == np.arange(k), dtype)

