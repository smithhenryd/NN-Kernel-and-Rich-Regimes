from model import Linear_Regression
import tensorflow as tf

## Dimension of input points
d = 20
## Number of training points
N = 10

# Create the training dataset:
## Recall that \beta parametrizes the diagonal neural network
beta = tf.random.normal([d, 1])

## Generate N training points [at random]
train_x = tf.random.uniform([N, d])
train_y = tf.reshape(tf.matmul(train_x, beta), (-1, 1))

# Train the diagonal neural network:
## Initialize the model
w0 = tf.ones([2*d, 1])
model = Linear_Regression(w0)

## Optimize using gradient descent with learning rate \eta = 10**(-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=10e-3)

## Train the diagonal nerual network using the mean squared error as our loss function L
MSE = tf.keras.losses.MeanSquaredError()
model.compile(optimizer, loss=MSE)
model.fit(train_x, train_y, epochs=10**4, verbose=0)

# Print the final training loss
tf.print(f"Training loss: {MSE(model.call(train_x), train_y)}")
tf.print(f"Weights: {model.trainable_weights[0]}")