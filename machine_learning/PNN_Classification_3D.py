import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
#locations=np.array(['J07', 'S01', 'W15'])
locations=np.array(['K04', 'O05', 'I08'])
index=np.concatenate([np.where(Y==i) for i in locations], axis=None)
dataTrainPlot=data_Train_shuffled[data_Train_shuffled.index.isin(index)]
#choose only b3002. b3003, b3005 columns
columns_select=np.array(['location', 'b3002', 'b3003', 'b3005'])
dataTrainPlot = dataTrainPlot[columns_select]
#transfer locations to color
classes_color, indices_color= np.unique (dataTrainPlot['location'] , return_inverse=True)
#print(indices_color)
xyz = plt.figure().add_subplot(111, projection='3d')
xyz.set_xlabel("b3002")
xyz.set_ylabel("b3003")
xyz.set_zlabel("b3005")
xyz.scatter(dataTrainPlot['b3002'], dataTrainPlot['b3003'], dataTrainPlot['b3005'], c=indices_color, label=locations)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
 #   print(dataTrainPlot)
plt.show()
dataTrain=dataTrainPlot[['b3002', 'b3003', 'b3005']]
dataTrain['indices_color']=indices_color
#print(dataTrain)
X = dataTrain[['b3002', 'b3003', 'b3005']]
Y = dataTrain['indices_color']
print(len(Y))
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.5, random_state=0)
dataTrain = pd.concat([Xtrain, Ytrain], axis=1, sort=False)
dataTrain=dataTrain.to_numpy()
print(dataTrain)

#print(dataTrain)
"""
data_Test = data_Test.drop('date', axis=1)
data_Test = data_Test.drop('location', axis=1)
data_Test_names=list(data_Test.columns)
x_scaled = min_max_scaler.fit_transform(data_Test)
data_Test = pd.DataFrame(x_scaled)
data_Test.columns=data_Test_names
print(data_Test)
dataTest=data_Test[['b3005', 'b3004', 'b3013']]
dataTest=dataTest.to_numpy()
"""
dataTest=Xtest.to_numpy()
print(dataTest)

#Using the previous three ibecones to seperate the data

#menghitung PDF
def PDF(test,train,s):
    return math.exp(-((((test[0]-train[0])**2)+((test[1]-train[1])**2)+((test[2]-train[2])**2))/(2*(s)**2)))


#Klasifikasi
def klasifikasi(dtest,dtrain,s):
    kelas = {0:0.0,1:0.0,2:0.0}
    hasil = []
    for test in dtest:
        for train in dtrain:
            kelas[int(train[3])] = kelas[int(train[3])] + PDF(test,train,s)

        hasil.append(max(kelas.items(), key=operator.itemgetter(1))[0])
        kelas = {0:0.0,1:0.0,2:0.0}
    return np.array(hasil)


# menghitung akurasi
def tepat():
    n = 0
    smooth = []
    persenan = []
    while n < 1:
        n += 0.05
        akurasi = klasifikasi(dataTrain, dataTrain, n)
        sum = 0
        for i in range(len(dataTrain)):
            cek = akurasi[i] == int(indices_color[i])
            if cek:
                sum += 1
        persen = float(sum) / len(dataTrain) * 100
        persenan.append(persen)
        smooth.append(n)
        print("smoothing:", n, ", akurasi:", persen)
    print("nilai smoothing terbaik:", smooth[persenan.index(max(persenan))], "dengan akurasi: ", max(persenan))
    plt.xlabel("nilai smoothing")
    plt.ylabel("akurasi (%)")
    plt.plot(smooth, persenan)
    plt.show()
    return smooth[persenan.index(max(persenan))]


if __name__ == '__main__':
    kelas = np.concatenate((dataTest, klasifikasi(dataTest, dataTrain, tepat())[:, None]), axis=1)

    # visualisasi hasil======
    abc = plt.figure().add_subplot(111, projection='3d')

    abc.set_xlabel("b3002")
    abc.set_ylabel("b3003")
    abc.set_zlabel("b3005")

    print(accuracy_score(Ytest, kelas[:, 3]))

    abc.scatter(kelas[:, 0], kelas[:, 1], kelas[:, 2], c=kelas[:, 3])
    plt.show()
    # visualisasi hasil======

    f = open('prediksi.txt', 'w')
    f.write("\n".join(map(lambda x: str(x), kelas)))
    f.close()

#print(X)
"""



np.random.seed(0)

pipeline = make_pipeline(
    LinearSVC()
)

pipeline.fit(Xtrain, Ytrain)
guesses = pipeline.predict(Xtest)
print(accuracy_score(Ytest, guesses))


#replace 1.00 to zero
classes, indices= np.unique (Y , return_inverse=True)
print(classes)
print(indices)

num = X._get_numeric_data()
num[num >=1.0] = 0
#print(X)


#Baseline
dummy = DummyClassifier('most_frequent')
dummy.fit(Xtrain, Ytrain)
print(cross_val_score(dummy, Xtrain, Ytrain, cv=5, scoring='accuracy').mean())

classifiers = [
    tree.DecisionTreeClassifier(),
    ensemble.RandomForestClassifier(),
    ensemble.GradientBoostingClassifier(),
    neural_network.MLPClassifier()
]
for item in classifiers:
    print(item)
    clf = item
    print(cross_val_score(clf, Xtrain, Ytrain, scoring='accuracy', cv=5).mean())

"""
