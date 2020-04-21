import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score
from pyGRNN import GRNN #imports the GRNN regressor module
from pyGRNN import feature_selection as FS #imports the GRNN feature selector module
import time
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
warnings.simplefilter('ignore')
warnings.filterwarnings(action='once')

data_Train = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Labeled.csv", sep=",")
data_Test = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Unlabeled.csv", sep=",")

#remove date column
data_Train = data_Train.drop(['date'], axis=1)
Y = data_Train['location']
classes_color, indices_color= np.unique(Y, return_inverse=True)
X = data_Train.drop('location', axis=1)
X['location']=indices_color
Y = X['location'].values.ravel()
X = X.drop('location', axis=1).values
#shuffle data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(preprocessing.minmax_scale(X), Y.reshape((-1, 1)), test_size=0.2, random_state = 42, shuffle=True)
featnames=list(data_Train.drop('location', axis=1).columns)
#print(X_names)
#normalization data X

#print(X)

pca = PCA(n_components=13, random_state=42)
X_pca = pca.fit_transform(Xtrain)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.plot(np.arange(13), pca.explained_variance_ratio_, linewidth  = 1.5, ls = '-')
plt.show()
sum=0
for i in range(len(pca.explained_variance_ratio_)):
    sum+=pca.explained_variance_ratio_[i]
    print(f'First {i} PCA can explain {sum}% data')

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(Xtrain[:,0], Xtrain[:,1], Ytrain[:,0], c=Ytrain[:,0], marker='o')
ax.set_xlabel('b3001')
ax.set_ylabel('b3002')
ax.set_zlabel('Y')
plt.show()
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(Xtrain[:,3], Xtrain[:,4], Ytrain[:,0], c=Ytrain[:,0], marker='o')
ax.set_xlabel('b3004')
ax.set_ylabel('b3005')
ax.set_zlabel('Y')
plt.show()
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(Xtrain[:,6], Xtrain[:,7], Ytrain[:,0], c=Ytrain[:,0], marker='o')
ax.set_xlabel('b3007')
ax.set_ylabel('b3008')
ax.set_zlabel('Y')
plt.show()
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(Xtrain[:,9], Xtrain[:,10], Ytrain[:,0], c=Ytrain[:,0], marker='o')
ax.set_xlabel('b3010')
ax.set_ylabel('b3011')
ax.set_zlabel('Y')
plt.show()


pd.plotting.scatter_matrix(data_Train, alpha=0.2, figsize=(20,20))
plt.savefig('features_relation.png')
plt.show()

IsotropicSelector = FS.Isotropic_selector()

start = time.time()
IsotropicSelector.relatidness(Xtrain, feature_names=featnames, strategy = 'ffs')
IsotropicSelector.plot_(featnames)

plt.show()
print('Time to complete the search [s]: ' + str(time.time() - start))


print('Selecting the best subset of features using a forward fs strategy:')
start = time.time()

IsotropicSelector.ffs(Xtrain, Ytrain, feature_names=featnames, stop_criterion='full_search')
print('Time to complete the search [s]: ' + str(time.time() - start))

print('Selecting the best subset of features using a backward fs strategy:')
start = time.time()
IsotropicSelector.bfs(Xtrain, Ytrain, feature_names=featnames,  stop_criterion='full_search')
print('Time to complete the search [s]: ' + str(time.time() - start))


print('Selecting the best subset of features using an exhaustive search:')
start = time.time()
IsotropicSelector.es(Xtrain, Ytrain, feature_names=featnames)
print('Time to complete the search [s]: ' + str(time.time() - start))

print('Performing a complete feature selection from scratch:')
start = time.time()
IsotropicSelector.feat_selection(Xtrain, Ytrain, feature_names=featnames, strategy = 'es')
print('Time to complete the feature selection [s]: ' + str(time.time() - start))
best_set = IsotropicSelector.best_inSpaceIndex
X_train_BestSet = Xtrain[:,best_set]
X_test_BestSet = Xtest[:,best_set]
# Instantiate the estimator
IGRNN = GRNN()
# Define the parameters for a GridSearch CV and fit the model
params_IGRNN = {'kernel':["RBF"],
                'sigma' : list(np.arange(0.1, 4, 0.01)),
                'calibration' : ['None']
                 }
grid_IGRNN = GridSearchCV(estimator=IGRNN,
                          param_grid=params_IGRNN,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=1,
                          n_jobs = -1
                          )
grid_IGRNN.fit(X_train_BestSet, Ytrain.ravel())
# Use the best model to perform prediction, and compute mse
best_model = grid_IGRNN.best_estimator_
Ypred_IGRNN = best_model.predict(X_test_BestSet)
mse_IGRNN = MSE(Ytest, Ypred_IGRNN)
Ypred_IGRNN=np.round(Ypred_IGRNN,0)
grid_IGRNN.fit(Xtrain, Ytrain.ravel())
best_model = grid_IGRNN.best_estimator_
Ypred_IGRNN_be=best_model.predict(Xtest)
mse_IGRNN_be = MSE(Ytest, Ypred_IGRNN_be)
Ypred_IGRNN_be=np.round(Ypred_IGRNN,0)
#print(accuracy_score(Ytest, Ypred_IGRNN))


AnisotropicSelector = FS.Anisotropic_selector()
start = time.time()
AnisotropicSelector.max_dist(Xtrain, Ytrain.ravel(), feature_names=featnames)
print('Time to complete the feature selection [s]: ' + str(time.time() - start))


AGRNN = GRNN()
AGRNN.fit(Xtrain, Ytrain.ravel())
sigma=AGRNN.sigma
Ypred_AGRNN = AGRNN.predict(Xtest)
mse_AGRNN = MSE(Ytest, Ypred_AGRNN)
Ypred_AGRNN=np.round(Ypred_AGRNN,0)
print('MSE with AGRNN (with "embedded" feature selection): ' + str(mse_AGRNN))
print('MSE with IGRNN (after feature selection): ' + str(mse_IGRNN))
print('MSE with IGRNN (before feature selection): ' + str(mse_IGRNN_be))
print(accuracy_score(Ytest,Ypred_AGRNN))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# create a color palette
palette = plt.get_cmap('Set1')
ax.set_yscale("log")
ax.set_xticklabels(['','b3001', 'b3002','b3003', 'b3004','b3005', 'b3006',  'b3007', 'b3008', 'b3009', 'b3010', 'b3011', 'b3012', 'b3013'])
ax.tick_params(labelsize=11)
plt.plot(np.arange(13), sigma, marker = 'v',markersize = 8, color = 'black')
ax.axhline(np.sqrt(8), color='red', alpha = 0.5,  linewidth  = 1.5, ls = '--')
ax.text(0.08,  3, r'Treshold = $\sqrt{p}$', style='italic', fontsize = 11)
plt.xlabel('Feature', fontsize=14)
plt.ylabel('Optimal bandwidth', fontsize=14)
plt.title('Optimal bandwidth for each feature, with CI',fontsize=16)
plt.ylim((0.001, 10))
plt.show()
