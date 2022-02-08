import tensorflow as tf
from ..Linear_NN.model import Linear_Regression

class Linearized_Model(tf.keras.Model):
    """
    The linearized version of a preexisting, D-homogeneous TensorFlow model, as discussed in
    Chizat et al. 2018 as well as in Woodworth et al. 2020
    """

    def __init__(self, model, train_x, alpha=1, hom=1, **kwargs):
        """
        Initializes the network, which is simply 'model' linearized around its initialization 'w0'

        model: a tensorflow.keras.Model object, representing the [nonlinear], D-homogeneous model
        for which one would like to compute the linearized model
        NOTE: the model must have current weight vector equal to its initialization (i.e. it should not be trained yet)
        
        train_x: an  N x d dimensional Tensor whose rows are the vectors with which the model is trained 
        alpha: a nonnegative float, representing the scale with which 'model' was initialized
        hom: a positive integer, the degree of homogeneity of 'model'
        """

        super(Linearized_Model, self).__init__(**kwargs)

        self.linearized_layer_1 = Linearized_Layer(model, train_x, alpha, hom, **kwargs)

    def call(self, train_x) -> tf.Tensor:
        """
        Evaluates the linearized model at the input data **with which the model is trained**
        NOTE: to evaluate the model at arbitrary input data, one should call self.evaluate()

        returns: an N x 1 dimensional Tensor containing the corresponding outputs of the model
        """

        return self.linearized_layer_1(train_x)

    def evaluate(self, model, inputs) -> tf.Tensor:
        """
        Evaluates the linearized model at **arbitrary input data**

        model: the tensorflow.keras.Model object representing the nonlinear model, should not yet be trained

        returns an M x 1 dimensional Tensor containing the corresponding outputs of the model
        """

        return self.linearized_layer_1.evaluate(model, inputs)

class Linearized_Layer(tf.keras.layers.Layer):
    """
    The single layer of the linearized network
    """

    def __init__(self, model, train_x, alpha, hom, **kwargs):
        """
        Initializes the linearized layer
        """

        super(Linearized_Layer, self).__init__(**kwargs)
        
        # Compute the bias of the model at w = w0 on the training data
        self.bias_scale = alpha**hom
        self.bias = model(train_x)

        # Compute the gradient of the model with respect to the weights w at w = w0, evaluated on the training data 
        self.grads = self._compute_gradients(model, train_x)

        # Initialization of nonlinear model
        self.init = tf.identity(model.trainable_weights) 

        # Initialize the weight vector [that which we will train]
        self.w = tf.Variable(initial_value=tf.reshape(model.trainable_weights, [-1, 1]), trainable=True)

    def _compute_gradients(self, model, inputs):
        """
        Computes the gradient of the nonlinear model with respect to the weight vector w at w = w0 evaluated on each input x
        """

        # Number of inputs
        N = int(tf.shape(inputs)[0])
        # Dimension of parameter space
        p = int(tf.shape(tf.reshape(model.trainable_weights, [-1, 1]))[0])

        with tf.GradientTape() as g:
            
            # Forward pass
            model_eval = model(inputs)

            # Calculate gradient at each input x
            grad = g.jacobian(model_eval, model.trainable_weights)
        
        return tf.reshape(grad, [N, p])

    def call(self, inputs) -> tf.Tensor:
        """
        Evaluates model at training data
        """
        #self.bias_scale*self.bias + tf.matmul(self.grads ,tf.reshape(self.w - self.init), [-1, 1])
        return self.bias_scale*self.bias + tf.matmul(self.grads,tf.reshape(self.w - self.init, [-1, 1]))
    
    def evaluate(self, model, inputs):
        """
        Evaluates the model at arbitrary input data
        """

        # Compute the bias term
        bias = model(inputs)

        # And the gradients
        grads = self._compute_gradients(model, inputs)

        return self.bias_scale*bias + tf.matmul(grads,tf.reshape(self.w - self.init, [-1, 1]))

class LinearizedCallback(tf.keras.callbacks.Callback):
    """
    Modifies a Linearized_Model object so that when training is complete, calling
    model (inputs) will evaluate the model on arbitrary [rather than training] inputs
    (this eliminates the need for the 'evaluate' method in the original object)
    """
    def __init__(self, model, **kwargs):
        """
        Initializes the callback object

        model: the nonlinear tensorflow.keras.Model object corresponding to the Linearized_Model
        """

        super(LinearizedCallback, self).__init__(**kwargs)
        
        self.nonlinear_model = model

    def on_train_end(self,logs=None):
       
        self.model.__call__ = lambda inputs: self.model.evaluate(self.nonlinear_model, inputs)
        self.model.call = self.model.__call__

if __name__ == "__main__":

    tf.random.set_seed(100)

    # Create the linear regression model (which is nonlinear in its weights w)
    d = 3
    w0 = tf.ones([2*d, 1])
    model = Linear_Regression(w0)

    # Instantiate the linearized model
    # The linear regression model is 2-homogeneous

    # Suppose we will train the model with N input vectors
    N = 5
    train_x = tf.random.normal([N, d])
    linearized_model = Linearized_Model(model, train_x, alpha=1, hom=2)
    
    tf.print(f"Training outputs: {linearized_model(train_x)}")

    # Evaluate the model at test points
    test_x = tf.random.normal([2, d])
    tf.print(f"Test outputs: {linearized_model.evaluate(model, test_x)}")

    # Check that the model's trainable weights are correct
    tf.print(f"Trainable weights: {linearized_model.trainable_weights}")