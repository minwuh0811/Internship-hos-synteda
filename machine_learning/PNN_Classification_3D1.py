import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import operator
from sklearn.model_selection import train_test_split
import PNN_Classification_moreD

data_Train = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Labeled.csv", sep=",")
data_Test = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Unlabeled.csv", sep=",")
#remove date column
data_Train = data_Train.drop('date', axis=1)
#shuffle data
data_Train_shuffled = data_Train.sample(frac=1.0, random_state=0).reset_index(drop=True)
column_names=list(data_Train_shuffled.columns)
Y = data_Train_shuffled['location']
X = data_Train_shuffled.drop('location', axis=1)
#normalization data X
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)
#concat two dataframes
data_Train_shuffled = pd.concat([Y, X], axis=1, sort=False)
data_Train_shuffled.columns=column_names
#data visualization most three locations
classes, count=np.unique(Y,return_counts=True)
count_sort_ind = np.argsort(-count)
#print(classes[count_sort_ind])
#print(count[count_sort_ind])
# K04, O05, I08 most three locations
locations=np.array(['J07', 'S01', 'W15'])
index=np.concatenate([np.where(Y==i) for i in locations], axis=None)
dataTrainPlot=data_Train_shuffled[data_Train_shuffled.index.isin(index)]
#choose only b3002. b3003, b3005 columns
columns_select=np.array(['location', 'b3005', 'b3004', 'b3013'])
dataTrainPlot = dataTrainPlot[columns_select]
#transfer locations to color
classes_color, indices_color= np.unique (dataTrainPlot['location'] , return_inverse=True)
#print(indices_color)
xyz = plt.figure().add_subplot(111, projection='3d')
xyz.set_xlabel("b3005")
xyz.set_ylabel("b3004")
xyz.set_zlabel("b3013")
xyz.scatter(dataTrainPlot['b3005'], dataTrainPlot['b3004'], dataTrainPlot['b3013'], c=indices_color, label=locations)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
 #   print(dataTrainPlot)
plt.show()
dataTrain=dataTrainPlot[['b3005', 'b3004', 'b3013']]
dataTrain['indices_color']=indices_color
#print(dataTrain)
X = dataTrain[['b3005', 'b3004', 'b3013']]
Y = dataTrain['indices_color']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=0)

data = {'x_train': np.array(Xtrain),
        'x_test': np.array(Xtest),
        'y_train': np.array(Ytrain),
        'y_test': np.array(Ytest)}


predictions = PNN_Classification_moreD.PNN(data)
PNN_Classification_moreD.print_metrics(data['y_test'], predictions)