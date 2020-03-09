# pip install tensorflow==2.0.0
# https://www.tensorflow.org/install/gpu#linux_setup

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model
# from tensorflow.keras import layers
#
# # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# slicer = 60000
# x_train, x_test = x_train[:slicer], x_test[:slicer]
# y_train, y_test = y_train[:slicer], y_test[:slicer]
# x_train, x_test = x_train.reshape([- 1, 784]), x_test.reshape([- 1, 784])
# x_train, x_test = x_train / 255.0, x_test / 255.0
# y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)
# # x_train = x_train[..., tf.newaxis]
# # x_test = x_test[..., tf.newaxis]
#
# train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train)).shuffle(10000).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(32)
#
# # for elem in train_ds.as_numpy_iterator():
#     # print(elem)
#
# latent_space = 2
#
# class MyModel(Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         # self.conv1 = Conv2D(32, 3, activation='relu')
#         # self.flatten = Flatten()
#         self.d1 = Dense(128, activation='relu')
#         self.d2 = Dense(32, activation='relu')
#         self.d3 = Dense(latent_space, activation='relu')
#         self.d4 = Dense(32, activation='relu')
#         self.d5 = Dense(128, activation='relu')
#         self.d6 = Dense(784, activation='relu')
#
#     def call(self, x):
#         # x = self.conv1(x)
#         # x = self.flatten(x)
#         x = self.d1(x)
#         x = self.d2(x)
#         x = self.d3(x)
#         x = self.d4(x)
#         x = self.d5(x)
#         x = self.d6(x)
#         return x
#
# # class Generator(Model):
# #     def __init__(self):
# #         super(Generator, self).__init__()
# #         self.d1 = Dense(32, activation='relu')
# #         self.d2 = Dense(128, activation='relu')
# #         self.d3 = Dense(784, activation='relu')
# #
# #     def set(self, d1, d2, d3):
# #         self.d1.set_weights(d1.get_weights())
# #         self.d2.set_weights(d2.get_weights())
# #         self.d3.set_weights(d3.get_weights())
# #
# #     def call(self, x, training=False):
# #         # x = self.conv1(x)
# #         # x = self.flatten(x)
# #         x = self.d1(x)
# #         x = self.d2(x)
# #         x = self.d3(x)
# #         return x
#
# model = MyModel()
# model.build((None, 784))
# model.summary()
# loss_object = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.Adam()
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')
#
#
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss = loss_object(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
# def test_step(images, labels):
#     predictions = model(images, training=False)
#     t_loss = loss_object(labels, predictions)
#
#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
#
# EPOCH = 1
# for epoch in range(EPOCH):
#     train_loss.reset_states()
#     train_accuracy.reset_states()
#     test_loss.reset_states()
#     test_accuracy.reset_states()
#
#     for images, labels in train_ds:
#         train_step(images, labels)
#     for test_images, test_labels in test_ds:
#         test_step(test_images, test_labels)
#
#     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#     print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
#
# a = tf.reshape(x_test[10], [1, 784])
# # print(model(a))
# # print(y_test[10])
# #
# # plt.imshow(tf.reshape(model(a), [28, 28]))
# # plt.show()
# # plt.imshow(x_test[10])
# # plt.show()
#
# def generator_builder(d1, d2, d3):
#     model = tf.keras.Sequential()
#     # l1 = layers.Dense(32, input_shape=(2,))
#     # l2 = layers.Dense(128, input_shape=(32,))
#     # l3 = layers.Dense(784)
#     model.add(d1)
#     model.add(d2)
#     model.add(d3)
#
#     # l1.set_weights(d1.get_weights())
#     # l2.set_weights(d2.get_weights())
#     # l3.set_weights(d3.get_weights())
#     # activation='relu'
#     # model.compile(optimizer, loss)
#     # model.fit
#     return model
#
# generator = generator_builder(model.d4, model.d5, model.d6)# Generator()
# # generator.build((None, 2))
# # generator.set(model.d4, model.d5, model.d6)
# # generator.summary()
# # in = tf.reshape(model(tf.reshape(np.random.rand(2), [1, 2]))
# # inX = tf.reshape(np.random.rand(800), [1, 800]) # np.random.rand(2)
# # result = model(inX)
# # print(result)
# # plt.imshow(result, [28, 28])
# # plt.show()
#
# for x in range(5):
#     for y in range(5):
#         # noise = tf.random.uniform([1, latent_space])
#         noise = tf.reshape([1/(x+1), 1/(y+1)], [1, 2])
#         generated_image = generator.predict(noise)
#         plt.imshow(tf.reshape(generated_image, [28, 28]))
#         plt.show()
#
#
#
#
#
# # class MyModel(Model):
# #     def __init__(self):
# #         super(MyModel, self).__init__()
# #         self.conv1 = Conv2D(32, 3, activation='relu')
# #         self.flatten = Flatten()
# #         self.d1 = Dense(128, activation='relu')
# #         self.d2 = Dense(10, activation='softmax')
# #
# #     def call(self, x):
# #         x = self.conv1(x)
# #         x = self.flatten(x)
# #         x = self.d1(x)
# #         return self.d2(x)
# #
# #
# #
# # model = MyModel()
# #
# # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# # optimizer = tf.keras.optimizers.Adam()
# #
# # train_loss = tf.keras.metrics.Mean(name='train_loss')
# # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# # test_loss = tf.keras.metrics.Mean(name='test_loss')
# # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# #
# #
# # def train_step(images, labels):
# #     with tf.GradientTape() as tape:
# #         predictions = model(images, training=True)
# #         loss = loss_object(labels, predictions)
# #     gradients = tape.gradient(loss, model.trainable_variables)
# #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
# #
# #     train_loss(loss)
# #     train_accuracy(labels, predictions)
# #
# # def test_step(images, labels):
# #     predictions = model(images, training=False)
# #     t_loss = loss_object(labels, predictions)
# #
# #     test_loss(t_loss)
# #     test_accuracy(labels, predictions)
# #
# # EPOCH = 5
# # for epoch in range(EPOCH):
# #     train_loss.reset_states()
# #     train_accuracy.reset_states()
# #     test_loss.reset_states()
# #     test_accuracy.reset_states()
# #
# #     for images, labels in train_ds:
# #         train_step(images, labels)
# #     for test_images, test_labels in test_ds:
# #         test_step(test_images, test_labels)
# #
# #     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
# #     print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
#
#
#
#
#
# ###############################################################"""
# # model = tf.keras.models.Sequential([
# #   tf.keras.layers.Flatten(input_shape=(28, 28)),
# #   tf.keras.layers.Dense(100, activation=tf.nn.eu),
# #   tf.keras.layers.Dropout(0.2),
# #   tf.keras.layers.Dense(10, activation='softmax')
# # ])
# #
# # model.compile(optimizer='adam',
# #               loss='sparse_categorical_crossentropy',
# #               metrics=['accuracy'])
# #
# # model.fit(x_train, y_train, epochs=1)
# # print(model.predict(x_train))
# # model.evaluate(x_test, y_test)
#
# # print(model.layers[0])
# # plt.imshow(x_train[0])
# # plt.show()
