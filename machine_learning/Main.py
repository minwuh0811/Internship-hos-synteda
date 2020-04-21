import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
import NN as bk
import dataCount as dC
np.set_printoptions(threshold=sys.maxsize)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
#Loading data
data = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Labeled.csv", sep=",")
img = mpimg.imread("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_Layout.png")
data = data.drop('date', axis=1)
data_shuffled = data.sample(frac=1.0, random_state=0).reset_index(drop=True)
Y = data_shuffled['location']
X = data_shuffled.drop('location', axis=1)/-200

classes, indices= np.unique (Y , return_inverse=True)
len(classes)
classes, count=np.unique(Y,return_counts=True)
X=X.T
print (np.asarray((classes, count)).T)
num_train=np.floor(data_shuffled.shape[0]*0.95)
num=data_shuffled.shape[0]
one_hot = bk.one_hot_matrix(indices, C = 105)
print(len(one_hot.T[0]))
Y=pd.DataFrame(data=one_hot)
Xtrain=X.loc[:, 0:num_train]
Xtest=X.loc[:,num_train:num]
Ytrain=Y.loc[:,0:num_train]
Ytest=Y.loc[:,num_train:num]
locals=np.asarray((classes, count))
color=np.array(locals[1])
color_norm = (color - color.mean()) / (color.max() - color.min())
area=np.floor(np.array((30 * color_norm.astype(float))**2))
parameter=dC.xytraj(locals[0])
parameter=pd.DataFrame.from_dict(parameter)
ibeacon=['F09','J04','N04', 'S04','J07', 'N07','S07', 'J10', 'D15', 'J15', 'N15','R15', 'W15']
ibeacon=pd.DataFrame.from_dict(dC.xytraj(ibeacon))
parameterx=np.array(parameter['x'])
parametery=np.array(parameter['y'])
fig, ax= plt.subplots()
imgplot = ax.imshow(img, aspect='auto', extent=(-3.7,32,22.2,-2.8), alpha=0.5, zorder=-1)
ax.scatter(parameterx,parametery,c=color,s=area,alpha=0.5)
ax.scatter(ibeacon['x'],ibeacon['y'],s=20,c="red", edgecolors='black')
ax.set_xlabel("A-W")
ax.set_ylabel("1-18")
ax.set_xlim([-2, 30])
ax.set_ylim([-3, 20])
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.grid( which ='both')
ax.invert_yaxis()
plt.show()



#MLC




#parameters = model(Xtrain, Ytrain, Xtest, Ytest, num_epochs=100000,print_accuracy=True)
