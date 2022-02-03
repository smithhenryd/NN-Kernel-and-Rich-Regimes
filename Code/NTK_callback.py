import tensorflow as tf

class NTKCallback(tf.keras.callbacks.Callback):
    """
    A custom callback function for evaluating the neural tangent kernel (NTK)
    on the training data during neural network optimization; for additional information
    on the neural tangent kernel, see Jacot et al. 2018
    """
    
    def __init__(self, training_data, step=1, **kwargs):
        """
        Initializes the callback object

        training_data: a two-dimensional Tensor representing the data with which the network is trained;
        NOTE: training observerations are assumed to be stored in the *rows* of the Tensor

        step: an integer; every 'step' epochs of training, we evaluate the NTK at the training data
        """
        
        super(NTKCallback, self).__init__(**kwargs)

        self.step = step

        # Compute number of training observations, dimension of parameter space
        self.num_training = int(tf.shape(training_data)[0])

        # Initialize the grid of training points
        self.training_grid = self._create_training_grid(training_data)
        
        # Initialize the grid of NTK evaluations (a list of Tensors)
        # TODO: can we store this on the model object?
        self.NTK_evals = []
    
    def _create_training_grid(self, training_data) -> tuple[tf.Tensor]:
        """
        Builds the grid on which to evaluate the gradient of the model

        training_data: a two-dimensional Tensor representing the data with which the network is trained
        return: a tuple of Tensors, the first of which stores is the x-coordinates of the grid, the second of which
        stores the y-coordinates
        """

        d = int(tf.shape(training_data)[1])

        # First build the grid of X points
        X = tf.repeat(training_data, repeats=self.num_training, axis=0)
        # And the Y points
        Y = tf.tile(training_data, [self.num_training, 1])

        # Check that out grids have the correct dimension: N^2 x d
        try:
            tf.debugging.assert_equal(tf.shape(X), tf.shape(Y))
            tf.debugging.assert_equal(tf.shape(X), tf.constant([self.num_training**2, d], dtype=tf.int32))
        except tf.errors.InvalidArgumentError:
            raise ValueError(f"Grid of training points does not have the correct dimension (should be {self.num_training**2} x {d})")
    
        return (X, Y)
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Evaluates the NTK of the model on the training data

        epoch: an integer, the epoch number of the model during training
        """

        if not epoch % self.step:

            # Get the dimension of the model's parameter space
            if not epoch:
                self.p = int(tf.shape(tf.reshape(self.model.trainable_weights, [-1, 1]))[0])

            # Get the gradients of the model with respect to w evaluated on the grid of sample points
            grad_x, grad_y = self._get_gradient(self.training_grid[0]), self._get_gradient(self.training_grid[1])
            
            # Compute the \ell_2 inner product of grad_x and grad_y
            result = tf.math.reduce_sum(tf.math.multiply(grad_x, grad_y), axis=1, keepdims=True) 
            
            # Reshape the result to an N x N matrix
            self.NTK_evals.append(tf.reshape(result, [self.num_training, self.num_training]))
            print(f"\nEvaluated NTK on epoch {epoch}.")
        return

    def _get_gradient(self, input) -> tf.Tensor:
        """
        Computes the gradient of the model with respect to the weight vector w evaluated at each input x
        """
    
        N = int(tf.shape(input)[0])

        with tf.GradientTape() as g:

            # Evaluate the model at the input tensor 
            model_eval = self.model(input)

            # Then compute the gradient of the model f with respect to the weights w, evaluated at each input x
            grad = g.jacobian(model_eval, self.model.trainable_weights)
            
            # Reshape the output Tensor
            return tf.reshape(grad, [N, self.p])