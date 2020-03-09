import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * 3.571 + 8.754

X = np.random.rand(50)
Y = []
for x in X:
    Y.append(f(x))

weights = tf.Variable(0.5)
bias = tf.Variable(0.5)
optimizer = tf.optimizers.SGD(0.01)
loss_function = tf.losses.MeanSquaredError()
EPOCH = 2500


def predict(x):
    return weights * x + bias

def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = predict(x)
        loss = loss_function(y, predictions)
    gradients = tape.gradient(loss, [weights, bias])
    optimizer.apply_gradients(zip(gradients, [weights, bias]))

def optimize():
    for epoch in range(EPOCH):
        train_step(X, Y)
        if epoch % 50 == 0:
            print(loss_function(predict(X), Y).numpy())
    print(weights.numpy())
    print(bias.numpy())

optimize()

plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(weights * X + bias), label='Fitted line')
plt.legend()
plt.show()
