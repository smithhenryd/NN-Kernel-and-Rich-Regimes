import tensorflow as tf

class WeightsCallback(tf.keras.callbacks.Callback):
    """
    A custom callback function for storing the weights of the model throughout training
    """

    def __init__(self, step=1, **kwargs):
        """
        Initializes the callback object
        step: a positive integer; every 'step' epochs of training, we store the weights of the model
        """      
        
        super(WeightsCallback, self).__init__(**kwargs)
        
        self.step = step
        self.weight_evals = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Stores the weights of the model at the current epoch

        epoch: an integer, the epoch number of the model during training
        """

        if not epoch % self.step:

            self.weight_evals.append(self.model.get_weights())
            print("\nStored model weights.")