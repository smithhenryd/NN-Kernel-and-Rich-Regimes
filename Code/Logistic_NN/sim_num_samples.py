from model import get_ReLU_NN, get_logistic_dataset, LogisticLoss

import numpy as np
import tensorflow as tf

def compute_classification_err(y_true, y_pred):

    pred_sign =  tf.math.multiply(tf.cast(y_true, dtype=y_pred.dtype), y_pred)
    pred_sign = tf.math.greater(pred_sign, tf.constant([0], dtype=pred_sign.dtype))
    pred_sign = tf.reshape(tf.cast(pred_sign, tf.float32), [-1])

    return 1 - tf.math.reduce_mean(pred_sign)


def run_simulation(d, n, num_trials=20, num_epochs=2e4, init_scale=0.1, num_units=10, num_test_samples=1e3):

    # Generate training, test data
    n = int(n)
    num_test_samples = int(num_test_samples)
    num_epochs = int(num_epochs)

    X_train, Y_train = get_logistic_dataset(num_samples=n, d=d)
    print(X_train.shape)
    X_test, Y_test = get_logistic_dataset(num_samples=num_test_samples, d=d)
    print(X_test.shape)

    # We will use the logistic loss function
    logloss = LogisticLoss()

    # List in which we store the test error for each of our num_trials models
    test_err = []

    # For each of num_trials trials
    for i in range(num_trials):

        # Instantiate the neural network
        # NOTE: the logistic loss does not include the \ell^2 regularization term
        NN = get_ReLU_NN(d=d, units=num_units, rw=init_scale, ru=init_scale, lambd=0)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        # Train the network
        NN.compile(optimizer, loss=logloss)
        NN.fit(X_train, Y_train, validation_data= (X_test, Y_test), epochs=num_epochs, verbose=0)

        # Get the classification error for the model on the test set
        predictions_test = NN(X_test)
        test_err.append(float(compute_classification_err(Y_test, predictions_test)))

    # Return the average over all the test errors
    return sum(test_err)/num_trials


def recurse_simulate(threshold, d, n, num_trials=20, num_epochs=2e4, init_scale=0.1, num_units=10, num_test_samples=1e3, nlow=10, nhigh=10e5):

    print(f"Trying n={n}")

    # Run the simulation on (d, n) = (d, n)
    sim_err = run_simulation(d=d, n=n, num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples)

    print(f"(d, n) = ({d}, {n}), test error= {sim_err}")
    
    # If model does achieve the desired test accuracy
    if (1 - sim_err) >= threshold:
        
        # If bound is good enough, return n
        if n/nlow <= 1.2:
            return n
        # Otherwise, look for n lower than the current n
        else:
            return recurse_simulate(threshold=threshold, d=d, n=int(0.5*n), num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples, nlow=nlow, nhigh=n)

    # If model does not achieve the desired test accuracy
    else: 
        # If bound is good enough, return n
        if nhigh/n <= 1.2:
            return n

        # Otherwise, look for n larger than the current n 
        else:
            return recurse_simulate(threshold=threshold, d=d, n=int(1.5*n), num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples, nlow=n, nhigh=nhigh)            

def sim_num_samples(threshold=0.6, d_list=[20, 40, 80, 160, 320, 640], num_trials=20, num_epochs=2e4, init_scale=0.1, num_units=10, num_test_samples=1e3, print_progress=True):
    """
    Computes an approximation of the smallest sample size n necessary to achieve test accuracy >= threshold 

    threshold: a float 0 > and  < 1, the desired test accuracy for the model
    d_list: a list of positive integers, contains the values for the dimension of the 
    input space; default = [20, 40, 80, 160, 320, 640]
    num_trials: a positive integer, the number of gradient descent trials to run for each 
    pair of input dimension, d, and number of samples, n; the test error corresponding to 
    (d, n) is computed by averaging over the individual test errors for all num_trials trials;
    default = 20
    num_epochs: a positive integer, the number of epochs to run for each gradient descent simulation;
    default = 2e4
    init_scale: a positive float, the standard deviation with which the weights of the ReLU neural network
    num_units: a positive integer, the number of hidden units in the single hidden layer of the neural network;
    default=10
    num_test_samples: a positive integer, the number of test samples used to compute the test error of the
    network; default = 1e3
    print_progress: a boolean, True if simulation progress should be printed to stdout, False otherwise;
    default = True

    return: a list whci
    """

    smallest_N = []
    num_d_vals = len(d_list)

    for i in range(num_d_vals):

        if print_progress:
            print(f"Starting trials for d={d_list[i]}")

        # Sample size with which we begin our experiment
        d = d_list[i] 
        n = int(20*np.log(d))

        opt_n = recurse_simulate(threshold=threshold, d=d, n=n, num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples)
        
        if print_progress:
             print(f"Smallest n which achieves test accuracy >= {threshold} for d={d}: {opt_n}")

        smallest_N.append(opt_n)

    return smallest_N

if __name__ == "__main__":
    sim_num_samples(d_list=[20], num_epochs=3000, num_trials=3)