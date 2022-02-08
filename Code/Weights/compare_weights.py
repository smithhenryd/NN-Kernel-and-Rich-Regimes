import tensorflow as tf
import numpy as np

from ..Linear_NN.model import Linear_Regression
from ..Linearized_Model.linearized_model import Linearized_Model, LinearizedCallback
from weights_callback import WeightsCallback

# Number of training points
N = 10
# Dimension of training points 
d = 20

# Initialization shape and scale
alpha = 0.1
w0 = tf.ones([2*d, 1])

# Generate beta by taking each coordinate to be an iid Unif(0,1) random variable
beta = tf.random.uniform([d, 1], dtype=tf.float32)

# As in Woodworth et al., suppose our training points are drawn from a d-dimensional
# standard multivariate normal distribution
train_x = np.random.multivariate_normal(np.zeros((d)), np.identity(d), size=N)
train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)

# Compute the corresponding y-values
train_y = tf.reshape(tf.matmul(train_x, beta), (-1, 1))

# Initialize our models
# Linear regression:
linrreg = Linear_Regression(w0, alpha=alpha)

# Linearized:
linrreg_const = Linear_Regression(w0, alpha=alpha)
# The linear regression model is 2 homogeneous
linearized = Linearized_Model(linrreg_const, train_x, alpha=alpha, hom=2)

# Number of epochs to train each network
epochs = 10**3

# Optimize each model using gradient descent
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

# With loss function equal to the mean-squared error
MSE = tf.keras.losses.MeanSquaredError()

callback_weights = WeightsCallback(10)
linrreg.compile(optimizer, loss=MSE)
linrreg.fit(train_x, train_y, epochs=10**3, verbose=1, callbacks=[callback_weights])

#callback = LinearizedCallback(linrreg_const)
#linearized.compile(optimizer, loss=MSE)
#linearized.fit(train_x, train_y, epochs=10**3, verbose=0, callbacks=[callback])

#tf.print(linrreg.linear_layer_1.w)
Wtf.print(linearized.linearized_layer_1.w)