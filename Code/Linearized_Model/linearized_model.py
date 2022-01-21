import tensorflow as tf
from ..Linear_NN.model import Linear_Regression

class Linearized_Model(tf.keras.Model):
    """
    The linearized version of a preexisting TensorFlow model, as discussed in
    Chizat et al. 2018 as well as in Woodworth et al. 2020
    """

    def __init__(self, model, **kwargs):
        """
        Initializes the network, which is simply 'model' linearized around its initialization 'w0'

        model: a tensorflow.keras.Model object, representing the [nonlinear] model
        for which one would like to compute the linearized model
        NOTE: the model must have current weight vector equal to its initialization (i.e. it should not be trained yet);
        if this is an issue, one should use copy.deepcopy or tf.keras.models.clone_model
        """

        super(Linearized_Model, self).__init__(**kwargs)

        self.linearized_layer_1 = Linearized_Layer(model, **kwargs)
    
    def call(self, inputs) -> tf.Tensor:
        """
        Evaluates the linearized model at the specified inputs

        inputs: a tuple containing (0) a M x d dimensional Tensor whose rows are the vectors at which the linearized model should
        be evaluated and (1) a tensorflow.keras.Model object, representing the [nonlinear] model

        returns: a M x 1 dimensional Tensor containing the corresponding outputs of the model
        """
        
        return self.linearized_layer_1(inputs)

class Linearized_Layer(tf.keras.layers.Layer):
    """
    The single layer of the linearized network
    """

    def __init__(self, model, **kwargs):
        """
        Initializes the linearized layer
        """

        super(Linearized_Layer, self).__init__(**kwargs)
        

        # Initialize the weight vector [that which we will train]
        self.w = tf.Variable(initial_value=tf.reshape(model.trainable_weights, [-1, 1]), trainable=True)


    def _compute_gradient(self, inputs, model):
        """
        Computes the gradient of the nonlinear model with respect to the weight vector w at w = w0 evaluated at each input x
        """

        with tf.GradientTape() as g:
            
            # Number of inputs
            N = int(tf.shape(inputs)[0])
            # Dimension of parameter space
            p = int(tf.shape(self.w)[0])

            # Forward pass
            model_eval = model(inputs)

            # Calculate gradient at each input x
            grad = g.jacobian(model_eval, model.trainable_weights)
            grad = tf.reshape(grad, [N, p])

            return grad

    def call(self, inputs) -> tf.Tensor:
        """
        Evaluates the linearized layer at the specified inputs
        """

        inputs, model = inputs
        
        # Compute the bias term
        bias = model(inputs)

        # As well as the gradient evaluated at the inputs
        grads = self._compute_gradient(inputs, model)

        return bias + tf.matmul(grads ,tf.reshape(self.w - tf.reshape(model.trainable_weights, [-1, 1]), [-1, 1]))

if __name__ == "__main__":

    tf.random.set_seed(100)

    # Create the linear regression model (which is nonlinear in its weights w)
    d = 3
    w0 = tf.ones([2*d, 1])
    model = Linear_Regression(w0)

    # Instantiate the linearized model
    linearized_model = Linearized_Model(model)
    
    # And evaluate it at N input vectors
    N = 5
    inputs = tf.random.normal([N, d])

    tf.print(f"Outputs: {linearized_model((inputs, model))}")
    tf.print(f"Trainable weights: {linearized_model.trainable_weights}")
