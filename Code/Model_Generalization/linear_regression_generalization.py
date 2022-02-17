import tensorflow as tf
import numpy as np
import pickle

from ..Linear_NN.model import Linear_Regression
from training_callbacks import LossThreshold, SaveLoss

def generate_sparse_dataset(N, d, r, indices=None, filename='training_data_sparse_regression.pk'):
    """
    Generates a training dataset corresponding to the overparameterized, r-sparse linear regression problem
    from Woodworth et al. 2020

    N: an integer, the number of total training points
    d: an integer, the dimension of the input space
    r: an integer, the number of nonzero coordinates in the beta vector with which the response values are generated;
    each nonzero entry in this vector has value 1/sqrt(r)
    indices: a list of integers, the coordinates of the beta vector which will be nonzero; by default, nonzero
    indices are chosen at random
    filename: a file name for saving the dataset; the X, Y, and beta arrays will be pickled as np.ndarray objects

    return: a tuple (X, Y, beta), where 
    X and Y are tf.Tensor objects representing input (N x d) and response (N x 1) data 
    beta is a tf.Tensor representing the vector with which the responses were generated
    """

    if N > d:
        raise ValueError(f"Dimension of the input space {d} must be at least as large as the number of training points {N}.")

    # Compute the beta vector
    beta = np.zeros(d)
    
    if not indices:
        indices = list(np.random.choice(d,size=r))

    if not r == len(indices):
        raise ValueError(f"Number of nonzero indices {r} must be equal to the length of indices list")

    for i in indices:
        beta[i] = 1/np.sqrt(r)

    # Want beta to be a column vector
    beta = np.reshape(beta, (d, 1))

    # Now generate the training points x_i according to a d-dimensional multivariate normal distribution
    X_train = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=N)
    assert X_train.shape == (N, d)

    # Generate the response points y_i according to a univariate normal distribution with mean \langle x_i, \beta \rangle
    # and variance 0.01
    Y_train = np.random.normal(loc=np.reshape(X_train@beta, (N)), scale=0.01, size=N)
    Y_train =np.reshape(Y_train, (N,1))
    assert Y_train.shape == (N,1)

    param_tuple = (X_train, Y_train, beta)
    
    # Pickle the training data (as np.ndarray objects) for posterity 
    with open(filename, 'wb') as f:
        pickle.dump(param_tuple, f)
    
    return tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(Y_train, dtype=tf.float32), tf.convert_to_tensor(beta, dtype=tf.float32)

def train_linreg_network(train_data, test_data, w0, alphas, lr, loss_threshold=10**(-4), max_epochs=10**4):
    """
    Trains a series of diagonal neural networks, as described in Woodworth et al. 2020, for each value of alpha

    train_data: a tuple of tf.Tensor objects representing the training data for the network, of the form (train_X, train_Y)
    test_data: a tuple of tf.Tensor objects representing the test data for the network, of the form (test_X, test_Y)
    w0: a tf.Tensor, the shape of the initialization
    alphas: a list of positive floats, the initialization scales with which the networks are trained
    lr: the learning rate (i.e. stepsize) for each of the networks
    loss_threshold: a nonnegative float; once the training loss is less than or equal to this value, training stops
    max_epochs: the maximum number of training epochs; after max_epochs, training will stop regardless of whether the loss threshold is reached
    """

    # Unpack training and test data
    train_x = train_data[0]
    train_y = train_data[1]

    test_x = test_data[0]
    test_y = test_data[1]

    print(f"Will train {len(alphas)} networks with initialization shape {w0}")

    # Store the training and test loss for each network trained
    train_losses = []
    test_losses = []

    # Store the number of epochs to converge
    num_epochs_conv = []

    # We will use the MSER as out loss function for all models
    MSE = tf.keras.losses.MeanSquaredError()

    for i in alphas:

        print(f"\n\nTraining network with alpha={i}\n\n")

        # Instantiate callback object
        model_loss = SaveLoss()
        model_threshold = LossThreshold(loss_threshold)

        # Instantiate the model with initialization scale i
        model = Linear_Regression(w0, i)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        # Compile the model
        model.compile(optimizer, loss=MSE)

        # And finally fit it
        model.fit(train_x, train_y, epochs=max_epochs, validation_data=[test_x, test_y],  callbacks=[model_loss, model_threshold])

        # Once the model has been fit, store the training and test loss 
        train_losses.append(model_loss.training_loss)
        test_losses.append(model_loss.test_loss)
        num_epochs_conv.append(model_threshold.epoch_count)

    return train_losses, test_losses, num_epochs_conv

if __name__ == "__main__":

    # Number of training points, dimension of training points
    num_training = 150
    d = 150 
    
    # Nonzero entries of beta vector
    indices = range(5)

    # Generate our training data
    training_data = generate_sparse_dataset(num_training, d, len(indices), indices=indices)

    # Initialization shape and scales
    # Let's start with only three alphas (and thus three networks)
    w0 = tf.ones([2*d, 1])
    alphas = [0.1, 1, 3]

    # Now let's create our test dataset
    num_test = 100
    test_data = generate_sparse_dataset(num_test, d, len(indices), indices=indices, filename='test_data_sparse_regression.pk')

    train_linreg_network((training_data[0], training_data[1]), (test_data[0], test_data[1]), w0, alphas=alphas, lr=10**(-3))