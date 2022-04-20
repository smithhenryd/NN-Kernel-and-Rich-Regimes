from model import get_ReLU_NN, get_logistic_dataset, LogisticLoss

import numpy as np
import tensorflow as tf
import pickle
import os

### NOTE: our implementation drew inspiration from that of Woodworth et al. 2020. ###
### We extend our gratitude to Dr. Woodworth for generously sending his code. ###

def compute_classification_err(y_true, y_pred):
    """
    Computes the classification error for the labels `y_true` and predictions `y_pred`
    """

    pred_sign =  tf.math.multiply(tf.cast(y_true, dtype=y_pred.dtype), y_pred)
    pred_sign = tf.math.greater(pred_sign, tf.constant([0], dtype=pred_sign.dtype))
    pred_sign = tf.reshape(tf.cast(pred_sign, tf.float32), [-1])

    return 1 - tf.math.reduce_mean(pred_sign)


def run_simulation(d, n, num_trials=20, num_epochs=2e4, init_scale=0.1, num_units=10, num_test_samples=1e3):
    """
    Approximates the population error for the neural network trained with n samples of input dimension d
    """

    # Generate training, test data
    n = int(n)
    num_test_samples = int(num_test_samples)
    num_epochs = int(num_epochs)

    X_train, Y_train = get_logistic_dataset(num_samples=n, d=d)
    X_test, Y_test = get_logistic_dataset(num_samples=num_test_samples, d=d)

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
    """
    Recursively approximates the test accuracy for a neural network trained with input dimension d;
    varies n to find the smallest n such that a test accuracy >= threshold is achieved
    """

    print(f"Trying n={n}")

    # Run the simulation on (d, n) = (d, n)
    sim_err = run_simulation(d=d, n=n, num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples)

    print(f"(d, n) = ({d}, {n}), test error= {sim_err}")
    
    # New n to try if simnulation does not achieve the desired accuracy
    # Note that this is the geometric mean of nlow and nhigh
    new_n = int(np.sqrt(nlow*nhigh))

    # If model does achieve the desired test accuracy
    if (1 - sim_err) >= threshold:
        
        # If bound is good enough, return n
        if n/nlow <= 1.2:
            return n
        # Otherwise, look for n lower than the current n
        else:
            new_n = int(np.sqrt(nlow*n))
            return recurse_simulate(threshold=threshold, d=d, n=new_n, num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples, nlow=nlow, nhigh=n)

    # If model does not achieve the desired test accuracy
    else: 
        # If bound is good enough, return n
        if nhigh/n <= 1.2:
            return n

        # Otherwise, look for n larger than the current n 
        else:
            new_n = int(np.sqrt(n*nhigh))
            return recurse_simulate(threshold=threshold, d=d, n=new_n, num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples, nlow=n, nhigh=nhigh)            

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
    print_progress: a boolean, True if simulation progress should be written to `progress.txt`, False otherwise;
    default = True

    return: a list containing an approximation of the smallest number of samples necessary to achieve
    the desired test accuracy
    """

    if print_progress:
        f = open("progress.txt", "w")
        f.write(f"sim_num_samples.py,\td={d_list},\tthreshold={threshold}:\n")
        f.write("*----*----*----*----*----*----*----*----*----*----*----*----*----*\n")
        f.write(f"Number of CPUs in use: {len(tf.config.list_physical_devices('CPU'))}\n")
        f.write(f"Number of GPUs in use: {len(tf.config.list_physical_devices('GPU'))}\n")
        f.write("*----*----*----*----*----*----*----*----*----*----*----*----*----*\n")
        f.flush()
        os.fsync(f.fileno())

    smallest_N = []
    num_d_vals = len(d_list)

    for i in range(num_d_vals):

        if print_progress:
            f.write(f"Starting trials for d={d_list[i]}\n")
            f.flush()
            os.fsync(f.fileno())

        # Sample size with which we begin our experiment
        d = d_list[i] 
        n = int(20*np.log(d))

        # Minimum and maximum sample size
        min_n = 10
        max_n= d**2

        opt_n = recurse_simulate(threshold=threshold, d=d, n=n, num_trials=num_trials, num_epochs=num_epochs, init_scale=init_scale, num_units=num_units, num_test_samples=num_test_samples, nlow=min_n, nhigh=max_n)
        
        if print_progress:
            f.write(f"Smallest n which achieves test accuracy >= {threshold} for d={d}: {opt_n}\n")
            f.flush()
            os.fsync(f.fileno())

        smallest_N.append(opt_n)

    if print_progress:
        f.write("*----*----*----*----*----*----*----*----*----*----*----*----*\n")
        f.close()
    return smallest_N

if __name__ == "__main__":
    
    smallest_N = sim_num_samples(threshold=0.6)

    with open('smallest_N_list.pk', 'wb') as f:
        pickle.dump(smallest_N, f)