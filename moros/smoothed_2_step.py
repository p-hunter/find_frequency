# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:09:21 2021

@author: phil.hunter
"""


import pandas as pd
from statsmodels.tsa import holtwinters
import numpy as np
import datetime as dt
from spectrum import *
from sklearn.linear_model import LinearRegression
import math

def smoothed_2_step(df, variable_name):
    
    
    """ 
    This model runs Holt-Winters exponential smoothing on a variable from the lead utilisation report
    it is then to be used in conjunction with a 2nd-stage model 
 
    """
    # Select only relevant cols
    df = df[["Date", variable_name]]
    
    # Extract some Date Features!
    df["Date"] = df["Date"].apply(lambda d: dt.datetime.strptime(d, "%d/%m/%Y"))
    df["DayOfTheWeek"] =df["Date"].apply(lambda d: dt.datetime.strftime(d, "%A"))
    df["DayOfTheMonth"] = df["Date"].dt.day
    
    df[variable_name] = df[variable_name].replace(np.nan, 0)
    
    # Detect the most prominent periodicity/seasonality
    s = find_frequency(df[variable_name])
    print("Seasonality is {}".format(s))
    
    
    hw_model = holtwinters.ExponentialSmoothing( 
        df[variable_name],
        trend=None,
        seasonal = "add",
        seasonal_periods = s
        ).fit()
    
    df["Resid"] = hw_model.resid
    
     
    
    
    return(df)




