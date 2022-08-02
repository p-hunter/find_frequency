# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:04:49 2021

@author: phil.hunter
"""

import pandas as pd
from statsmodels.tsa import holtwinters
import numpy as np
import datetime as dt
from spectrum import *
from sklearn.linear_model import LinearRegression
import math

def spec_yw(x,o):
    x_acf = acf(x, nlags = o)
    if x_acf[0] == 0:
        print("Zero Variance Found")
        return(0)    
    return