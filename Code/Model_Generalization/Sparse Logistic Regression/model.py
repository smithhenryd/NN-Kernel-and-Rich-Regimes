import tensorflow as tf 
import numpy as np

def get_ReLU_NN(d, units, rw =1, ru=1, lambd=0)->tf.keras.Model:
    """
    Initializes a two-layer ReLU neural network as discussed in Wei, Lee, Liu, and Ma 2020

    d: a positive integer, the dimension of the input space 
    units: a positive integer, the number of units (neurons) in the single hidden layer
    rw: a nonnegative float, the standard deviation of the initialization for the weights w_j in
    the output layer, default = 1
    ru: a nonnegative float, the standard deviation for each kth entry of the initialization for 
    the weights u_i in the hidden layer (that is, the standard deviation of (u_i)_k in the initialization),
    default = 1
    lambd: a nonnegative float, the parameter which determines the degree of \ell^2 regularization for the
    weights w_j and u_j, default = 0 (no \ell^2 regularization)
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
    model.add(tf.keras.layers.Dense(units=units, input_shape=(d,), activation="relu", use_bias=False, kernel_initializer=init_u, kernel_regularizer=tf.keras.regularizers.L2(lambd)))

    # Lastly, add the output layer with weights drawn from a N(0, rw^2) distribution
    init_w = np.random.normal(loc=0, scale=rw, size=(units, 1))
    init_w = tf.constant_initializer(value=init_w)
    model.add(tf.keras.layers.Dense(units=1, activation=None, use_bias=False, kernel_initializer=init_w, kernel_regularizer=tf.keras.regularizers.L2(lambd)))
    return model

def get_logistic_dataset(num_samples, d)->[tf.Tensor, tf.Tensor]:
    """
    Constructs a dataset drawn from the distribution specified by Wei, Lee, Liu, and Ma 2020
    
    The first two dimensions take values in x_i \in {0, 1} that encode the output y_i \in {-1, 1}, 
    the remaining d-2 input dimensions are random {-1, 1} bits i.e. noise

    num_samples: a positive integer, the number of samples in the dataset
    d: a positive integer > 2, the dimension of the input space
    """

    # The first two coordinates of the input data
    true_x = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    
    # The true labels associated with the input data
    true_y = np.array([1, 1, -1, -1])

    if not d > 2:
        raise ValueError(f"Dimension of dataset {d} must be > 2.")
    
    # Sample a point (x, y) from the distribution
    vals = np.random.randint(0, 4, size=num_samples)
    X = true_x[vals]
    Y = np.reshape(true_y[vals], (num_samples, 1))

    # The last d-2 coordinates of each input point is a random bit {-1, 1}
    X = np.concatenate((X, np.random.choice([-1, 1], size=(num_samples, d-2))), axis=1)

    return X, Y

class LogisticLoss(tf.keras.losses.Loss):
    """
    The logistic loss function for data (x_i, y_i), where the labels y_i \in {1, -1}

    Note that whenever y_i*f(x_i) > 0, then the loss is small; whenever y_i*f(x_i) <= 0,
    the loss is large
    """
    def __init__(self, **kwargs):
        
        super(LogisticLoss, self).__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        """
        Implements the logistic loss for labels y_true and predictions y_pred

        \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i f^{\text{NN}}(x_i; \Theta)))
        """

        # Convert y_true to a float
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        # And compute the loss as specified above
        loss = tf.math.log(1 + tf.math.exp((-1)*tf.multiply(y_true, y_pred)))
        return tf.reduce_mean(tf.reshape(loss, [-1]))

class ClassificationCallback(tf.keras.callbacks.Callback):
    """
    A callback function, stores the classification error of the network for {1, -1} labels
    at each epoch of training
    """

    def __init__(self, train_set, test_set=None, **kwargs):
        """
        train_set: a tuple of tf.Tensor objects, (1) an N x d dimensional tensor whose rows contain the input points of the training set,
        (2) an N x 1 dimensional tensor whose rows contain the ouputs (labels) of the training dataset
        train_set: a tuple of tf.Tensor objects, with the same format as train_set (the number of training observations N and test observations M
        need not be equal), by default None (no test data)
        """

        super(ClassificationCallback, self).__init__(**kwargs)

        self.x_train = train_set[0]
        self.y_train = train_set[1]
        self.train_err = []

        if test_set:
            self.x_test = test_set[0]
            self.y_test = test_set[1]
            self.test_err = []
        else:
            self.x_test = None
            self.y_test = None

    def on_epoch_end(self, epoch, logs=None):

        train_err_epoch = self._compute_classification_error(self.y_train, self.model(self.x_train))
        self.train_err.append(train_err_epoch)
        if self.x_test is not None:    
            test_err_epoch = self._compute_classification_error(self.y_test, self.model(self.x_test))
            self.test_err.append(test_err_epoch)
            print(f"\n Classification Error: {train_err_epoch} (Training), {test_err_epoch} (Test)")
        else:
            print(f"\n Classification Error: {train_err_epoch} (Training)")

    
    def _compute_classification_error(self, y_true, y_pred):
        """
        Computes the classification error for labels y_true \in {1, -1} and predictions y_pred
        """

        pred_sign =  tf.math.multiply(tf.cast(y_true, dtype=y_pred.dtype), y_pred)
        pred_sign = tf.math.greater(pred_sign, tf.constant([0], dtype=pred_sign.dtype))
        pred_sign = tf.reshape(tf.cast(pred_sign, tf.float32), [-1])
        return 1 - tf.math.reduce_mean(pred_sign)

if __name__ == "__main__":

    # Suppose we have 5-dimensional inputs and want to create a NN with 5 neurons in the hidden layer
    d = 20
    units = 10

    # Initialize ReLU NN with r_w = r_u = 0.1
    NN = get_ReLU_NN(d, units, rw=0.1, ru=0.1, lambd=0)

    # Print the model summary
    print(NN.summary())

    # And print the network weights
    print(NN.weights)

    # Finally, let's try evaluating the network at a sample point
    print(NN(tf.ones(shape=[1,d])))

    # Get 20 training points sampled from the distribution in Wei, Lee, Liu, and Ma 2020
    X_train, Y_train = get_logistic_dataset(20, d)
    X_test, Y_test = get_logistic_dataset(1000, d)

    # Optimize the network using gradient descent with stepsize 0.1
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)
    logloss= LogisticLoss()

    # Store the training and test classification error at each epoch of training
    mycallback = ClassificationCallback((X_train, Y_train), (X_test, Y_test))
    mycallback._compute_classification_error((X_train, Y_train))

    # Finally, compile and fit the model
    NN.compile(optimizer, loss=logloss)
    NN.fit(X_train, Y_train, validation_data= (X_test, Y_test), epochs=10**3, callbacks=[mycallback])