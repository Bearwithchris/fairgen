# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:13:34 2021

@author: Chris
"""

import numpy as np
import os

sample_moments= os.path.join("./samples/celeba_90_10_perc0.5_impweight/58","samples.npz")
test=np.load(sample_moments,'r')['x']