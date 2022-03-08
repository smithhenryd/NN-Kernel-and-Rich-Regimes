import tensorflow as tf 
import numpy as np

def get_ReLU_NN(d, units, rw =1, ru=1)->tf.keras.Model:
    """
    Initializes a two-layer ReLU neural network as discussed in Wei, Lee, Liu, and Ma 2020

    d: a positive integer, the dimension of the input space 
    units: a positive integer, the number of units (neurons) in the single hidden layer
    rw: a nonnegative float, the standard deviation of the initialization for the weights w_j in
    the output layer
    ru: a nonnegative float, the standard deviation for each kth entry of the initialization for 
    the weights u_i in the hidden layer (that is, the standard deviation of (u_i)_k in the initialization)
    """

    # Create a TensorFlow model
    model = tf.keras.Sequential()

    u = []
    for i in range(units):
        # For each unit in the hidden layer, the vector u_j is initalized from a multivariate normal distribution
        u.append(np.random.multivariate_normal(mean=np.zeros(d), cov=(ru**2)*np.eye(d)))

    # Now stack these u's into a weights matrix
    init_u = np.vstack(u)

    # Initialize the hidden layer in the network with this weight matrix
    init_u = tf.constant_initializer(value=np.transpose(init_u))
    model.add(tf.keras.layers.Dense(units=units, input_shape=(d,), activation="relu", use_bias=False, kernel_initializer=init_u))

    # Lastly, add the output layer with weights drawn from a N(0, rw^2) distribution
    init_w = np.random.normal(loc=0, scale=rw, size=(units, 1))
    init_w = tf.constant_initializer(value=init_w)
    model.add(tf.keras.layers.Dense(units=1, activation=None, use_bias=False, kernel_initializer=init_w))
    return model

if __name__ == "__main__":

    # Suppose we have two-dimensional inputs and want to create a NN with 5 neurons in the hidden layer
    d = 2
    units = 5
    NN = get_ReLU_NN(d, units)

    # Print the model summary
    print(NN.summary())

    # And print the network weights
    print(NN.weights)

    # Finally, let's try evaluating the network at a sample point
    print(NN(tf.ones(shape=[1,d])))