import tensorflow as tf
import numpy as np

class NTKCallback(tf.keras.callbacks.Callback):
    """
    A custom callback function for evaluating the neural tangent kernel (NTK)
    on the training data during neural network optimization; for additional information
    on the neural tangent kernel, see Jacot et al. 2018
    """
    
    def __init__(self, training_data, **kwargs):
        """
        Initializes the callback object

        training_data: a two-dimensional tensor representing the data with which the network is trained;
        note: training observerations are assumed to be stored in the *rows* of the tensor
        """
        
        super(NTKCallback, self).__init__(**kwargs)
        self.training_grid = self._create_training_grid(training_data)


    def _create_training_grid(self, training_data) -> tuple[tf.Tensor]:
        """
        Builds the grid on which to evaluate the gradient of the model

        training_data: a two-dimensional tensor representing the data with which the network is trained
        return: a tuple of Tensors, the first of stores is the x-coordinates of the grid, the second of which
        stores the y-coordinates
        """

        N = int(tf.shape(training_data)[0])
        d = int(tf.shape(training_data)[1])

        # First build the grid of X points
        X = tf.repeat(training_data, repeats=N, axis=0)
        # And the Y points
        Y = tf.tile(training_data, [N, 1])

        # Check that out grids have the correct dimension: N^2 x d
        try:
            tf.debugging.assert_equal(tf.shape(X), tf.shape(Y))
            tf.debugging.assert_equal(tf.shape(X), tf.constant([N**2, d], dtype=tf.int32))
        except tf.errors.InvalidArgumentError:
            raise ValueError(f"Grid of training points does not have the correct dimension (should be {N**2} x {d})")
    
        return (X, Y)
    
    def on_epoch_end(self, epoch, logs=None):
        self._create_training_grid
