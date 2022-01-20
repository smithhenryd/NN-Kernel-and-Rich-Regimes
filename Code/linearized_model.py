from re import L
import tensorflow as tf

class Linearized_Model(tf.keras.Model):
    """
    The linearized version of a preexisting TensorFlow model, as discussed in
    Chizat et al. 2018 as well as in Woodworth et al. 2020
    """

    def __init__(self, model, w0, **kwargs):
        """
        Initializes the network, which is simply 'model' linearized around its 
        initialization 'w0'

        model: a tensorflow.keras.Model object, representing the [nonlinear] model
        for which one would like to compute the 
        NOTE: the model must implement the __call__ method

        w0: the initialization of the tensorflow.keras.Model
        """

        super(Linearized_Model, self).__init__(**kwargs)
        
    
    def call():
        pass

class Linearized_Layer(tf.keras.layers.Layer):

    def __init__(self, model, w0, **kwargs):
        
        # Save the nonlinear model and its initialization
        self.nonlinear_model = model
        self.nonlinear_init = w0
        
        # Precompute the bias term
        self.bias = model(self.nonlinear_init)

        # And initialize the weight vector [to train]
        self.w = tf.Variable(initial_value=self.nonlinear_init, trainable=True)
    
    def _compute_gradient(self, inputs):


    def call(self, inputs) -> tf.Tensor:
        
        


