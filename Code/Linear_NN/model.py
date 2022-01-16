import tensorflow as tf 

class Linear_Regression():
    pass

class Linear(tf.keras.layers.Layer):
    """
    The single layer of the diagonal neural network
    """

    def __init__(self, w0, **kwargs):
        """
        Initializes the linear layer 

        w0: a 1 x 2d dimensional tensor, representing the initial weights for the linear layer
        """

        super(Linear, self).__init__(**kwargs)
        
        # Initialize the weight vector
        self.w = tf.Variable(initial_value=w0, trainable=True)
        self.w = tf.reshape(self.w, (1,-1))
        
        return
    
    def call(self, inputs):
        """
        Evaluates the linear layer at the specified inputs

        inputs: a d x N dimensional tensor whose columns contain the vectors at which the layer is to be evaluated
        
        return: a 1 x N dimensional tensor containing the corresponding outputs of the linear layer
        """

        weights_sq = tf.math.square(self.w)
        
        try:
            d = (tf.shape(self.w)[1])/2
            W = weights_sq[0:d] - weights_sq[d:]

            return tf.matmul(W, inputs)

        except:
            raise ValueError(f"Size of network weights must be twice the size of inputs: network weights {tf.shape(self.w)[1]}, inputs {tf.shape(inputs)[0]}")