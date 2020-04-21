import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
import NN as bk
data = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Labeled.csv", sep=",")
data_shuffled = data.sample(frac=1.0, random_state=0).reset_index(drop=True)
data = data_shuffled['location']
def xytraj(data):
    parameter={}
    x=[]
    y=[]
    for d in data:
        x.append(ord(d[0])-64.5)
        y.append(int(d[1:])+0.5)
    parameter['x']=x
    parameter['y']=y
    return parameter



