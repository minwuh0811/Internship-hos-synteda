import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import NN as bk

#Loading data
data = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Labeled.csv", sep=",")
unlabel = pd.read_csv("..\\BLE_RSSI_dataset\\BLE_RSSI_dataset\\iBeacon_RSSI_Unlabeled.csv", sep=",")
data = data.drop('date', axis=1)
data_shuffled = data.sample(frac=1.0, random_state=0).reset_index(drop=True)
Y = data_shuffled['location']
X = data_shuffled.drop('location', axis=1)/-200
classes, indices= np.unique (Y , return_inverse=True)
X=X.T
num_train=np.floor(data_shuffled.shape[0]*0.8)
num=data_shuffled.shape[0]
one_hot = bk.one_hot_matrix(indices, C = 105)
Y=pd.DataFrame(data=one_hot)
Xtrain=X.loc[:, 0:num_train]
Xtest=X.loc[:,num_train:num]
Ytrain=Y.loc[:,0:num_train]
Ytest=Y.loc[:,num_train:num]

parameters=bk.initialize_parameters()

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.02,
          num_epochs=50000, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    costs = []  # To keep track of the cost
    train_val=[]
    test_val=[]
    # Initialize parameters
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')
    parameters = bk.initialize_parameters()
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = bk.forward_propagation(X, parameters)
    # Cost function: Add cost function to tensorflow graph
    cost = bk.compute_cost(Z3, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initialize all the variables
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            seed = seed + 1
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
            _, batch_cost = sess.run([optimizer, cost], feed_dict={X: Xtrain, Y: Ytrain})
            epoch_cost += batch_cost
            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
            correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            train_val.append(accuracy.eval({X: X_train, Y: Y_train}))
            test_val.append(accuracy.eval({X: X_test, Y: Y_test}))
# plot the cost
        plt.subplot(121)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.legend()
        plt.subplot(122)
        plt.plot(np.squeeze(train_val), label='train_val',color='blue')
        plt.plot(np.squeeze(test_val),label='test_val', color='red')
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters



parameters = model(Xtrain, Ytrain, Xtest, Ytest, num_epochs=30000)
