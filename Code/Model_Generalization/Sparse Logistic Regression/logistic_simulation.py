import tensorflow as tf
import numpy as np
import pickle

from model import get_ReLU_NN, get_logistic_dataset, LogisticLoss, ClassificationCallback

# Number of simulations
N = 10

# Dimension of input space
d = 20

# Number of hidden units in ReLU NN
units = 10

# Get our training, test data
try:
    with open('training_data_logistic_200.pk', 'rb') as f:
        X_train, Y_train = pickle.load(f)

except FileNotFoundError:
    X_train, Y_train = get_logistic_dataset(200, d)
    with open('training_data_logistic_200.pk', 'wb') as f:
        pickle.dump((X_train, Y_train), f)

try:
    with open('test_data_logistic.pk', 'rb') as f:
        X_test, Y_test = pickle.load(f)

except FileNotFoundError:
    X_test, Y_test = get_logistic_dataset(1000, d)
    with open('test_data_logistic.pk', 'wb') as f:
        pickle.dump((X_test, Y_test), f)

logloss= LogisticLoss()

train_err_arrays = []
test_err_arrays = []
weights = []

for i in range(N):

  print(f"**************** \n Iteration {i+1}/{N} \n****************")

  NN = get_ReLU_NN(d, units, rw=1, ru=1, lambd=0)
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    
  mycallback = ClassificationCallback((X_train, Y_train), (X_test, Y_test))
  NN.compile(optimizer, loss=logloss)

  NN.fit(X_train, Y_train, validation_data= (X_test, Y_test), epochs=2*(10**4), callbacks=[mycallback])
  
  train_err_arrays.append(mycallback.train_err)
  test_err_arrays.append(mycallback.test_err)

  NN_weights = [i.numpy() for i in NN.weights]
  weights.append(NN_weights)

with open('train_err_simulations_200.pk', 'wb') as f:
        pickle.dump(train_err_arrays, f)

with open('test_err_simulations_200.pk', 'wb') as f:
        pickle.dump(test_err_arrays, f)

with open('network_weights_simulations_200.pk', 'wb') as f:
        pickle.dump(weights, f)