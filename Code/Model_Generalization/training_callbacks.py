import tensorflow as tf

class LossThreshold(tf.keras.callbacks.Callback):
    """
    A Callback object for early stopping; training is stopped once
    the network has training error below the specified threshold
    """

    def __init__(self, threshold, **kwargs):
      
      super(LossThreshold, self).__init__(**kwargs)
      self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        
      loss = logs["loss"]
      
      if loss <= self.threshold:
        self.epoch_count = epoch
        self.model.stop_training = True
    
class SaveLoss(tf.keras.callbacks.Callback):
    """
    A Callback object for storing the network's training and test loss throughout training;
    at the end of each epoch the training and test loss are appended to the respective lists
    """
    
    def __init__(self, **kwargs):

      super(SaveLoss, self).__init__(**kwargs)
      self.training_loss = []
      self.test_loss = []
    
    def on_epoch_end(self, batch, logs=None):
      self.training_loss.append(logs['loss'])
      self.test_loss.append(logs['val_loss'])