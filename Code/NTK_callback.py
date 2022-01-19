import tensorflow as tf

class NTKCallback(tf.keras.callbacks.Callback):
    """
    A custom callback function for evaluating the neural tangent kernel (NTK)
    on the training data during neural network optimization; for additional information
    on the neural tangent kernel, see Jacot et al. 2018

    NOTE: The TensorFlow model to which the callback function binds must implement the __call__ method
    """
    
    def __init__(self, training_data, step=1, **kwargs):
        """
        Initializes the callback object

        training_data: a two-dimensional Tensor representing the data with which the network is trained;
        note: training observerations are assumed to be stored in the *rows* of the Tensor
        step: an integer; every 'step' epochs of training, we evaluate the NTK at the training data
        """
        
        super(NTKCallback, self).__init__(**kwargs)

        self.step = step
        self.num_training = int(tf.shape(training_data)[0])
        # Initialize the grid of training points
        self.training_grid = self._create_training_grid(training_data)
        # Initialize the grid of NTK evaluations (a list of Tensors)
        # TODO: can we store this on the model object?
        self.model.NTK_evals = []


    def _create_training_grid(self, training_data) -> tuple[tf.Tensor]:
        """
        Builds the grid on which to evaluate the gradient of the model

        training_data: a two-dimensional Tensor representing the data with which the network is trained
        return: a tuple of Tensors, the first of stores is the x-coordinates of the grid, the second of which
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

        if not epoch % self.step:
            
            # Matrix containing the evaluations of the NTK
            NTK_matrix = tf.zeros([self.num_training, self.num_training], dtype=tf.float32)
            
            for i in range(self.num_training):
                for j in range(self.num_training):
                    
                    # Training points at which to evaluate the gradient of the model wrt weights
                    index = i*self.num_training + j
                    x, y = self.training_grid[0][index,:], self.training_grid[1][index,:]
                    grad_x, grad_y = self._get_gradient(x), self._get_gradient(y)
                    
                    # Compute the \ell_2 inner product of grad_x and grad_y
                    NTK_matrix[i,j] = tf.tensordot(grad_x, grad_y, 2)
            
            # Append the neural tangent kernel matrix to the list of evaluations
            self.model.NTK_evals(NTK_matrix)
        return
    
    def _get_gradient(self, x) -> tf.Tensor:
        """
        Evaluates the gradient of the model with respect to the trainable weights at the point x

        x: a 1 x d dimensional Tensor, where d is the dimension of the input to the model
        return: a 1 x p dimensional Tensor, where p is the dimension of the parameter space
        """

        with tf.GradientTape() as g:
            
            # TODO: Not sure if this line is needed (we are not computing the gradient wrt x)
            g.watch(x)

            # Evaluate the model at x 
            model_eval = self.model(x)

            # Then compute the gradient of the model f with respect to the weights w, evaluated at the point x
            return g.gradient(model_eval, self.model.trainable_weights)      