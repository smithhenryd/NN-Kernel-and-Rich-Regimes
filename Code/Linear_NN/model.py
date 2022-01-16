import tensorflow as tf 

class Linear_Regression(tf.keras.Model):
    """
    The diagonal neural network considered in Woodworth et al. 2020
    Note that this is just simple linear regression problem with an alternate
    parameterization of the weights
    """

    def __init__(self, w0, alpha=1, **kwargs):
        """
        Initializes the linear regression model

        w0: a 1 x 2d dimensional tensor, representing the "shape" of the initialization
        alpha: a nonnegative float, representing the multiplicative factor by which w0 is scaled
        """

        super(Linear_Regression, self).__init__(**kwargs)

        # The network contains a single linear layer
        self.linear_layer_1 = Linear(alpha*w0, **kwargs)

    def call(self, inputs):
        """
        Evaluates the linear regression model at the specified inputs

        inputs: a d x N dimensional tensor whose columns contain the vectors at which the model is to be evaluated
        
        return: a 1 x N dimensional tensor containing the corresponding outputs of the model
        """
        return self.linear_layer_1(inputs)



class Linear(tf.keras.layers.Layer):
    """
    The single layer of the diagonal neural network
    """

    def __init__(self, w0, **kwargs):
        """
        Initializes the linear layer 
        """

        super(Linear, self).__init__(**kwargs)
        
        # Initialize the weight vector
        self.w = tf.Variable(initial_value=w0, trainable=True)
        self.w = tf.reshape(self.w, (1,-1))
            
    def call(self, inputs):
        """
        Evaluates the linear layer at the specified inputs
        """

        weights_sq = tf.math.square(self.w)
        try:
            d = int(tf.shape(self.w)[1]/2)
            W = tf.reshape(weights_sq[0,0:d] - weights_sq[0,d:], (1, -1))
            return tf.matmul(W, inputs)

        except:
            raise ValueError(f"Size of network weights must be twice the size of inputs: network weights {tf.shape(self.w)[1]}, inputs {tf.shape(inputs)[0]}")

if __name__ == '__main__':
    
    # Create the linear regression model
    d = 3
    w0 = tf.ones([1, 2*d])
    model = Linear_Regression(w0)

    # And evaluate it at N test points
    N = 5
    inputs = tf.random.normal([d, N])
    outputs = model.call(inputs)

    # Verify f(w0) = 0
    assert tf.norm(tf.zeros([N, 1]) - outputs) < 1e-10