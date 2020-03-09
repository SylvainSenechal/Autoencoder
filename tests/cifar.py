import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# TODO: voir expand dim pour reshape
# TODO: voir utilite validation data
# TODO: utilise model.predict :
# pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
# TODO: voir difference entre loss et val_loss, accuracy et val accuracy

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x, test_x = train_x / 255.0, test_x / 255.0
# NB : pour image non rgb, il va manquer une dimension (mnist, (60000, 28, 28)), faire un reshape :
train_x, test_x = train_x.reshape(train_x.shape[0], 28, 28, 1), test_x.reshape(test_x.shape[0], 28, 28, 1)


model = models.Sequential()
print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
print(train_x.shape)
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# 896 params => 32 filtres * filtres de tailles 3 * 3 * rgb = 3 + 1 bias par filtre <=> 32 * 3 * 3 * 3 + 32 = 896
# output : 32 images (30*30)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 18496 params => 64 filtres * 3 * 3 * profondeur de 32 + bias <=> 64 * 3 * 3 * 32 + 64 =18496
# output : 64 images (13*13)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 36928 params => 64 filtres * 3 * 3 * profondeur de 64 + bias <=> 64 * 3 * 3 * 64 + 64 = 36928
# output : 64 images (4*4)

model.add(layers.Flatten())
# output : vecteur 1d de taille 64 * 4 * 4

model.add(layers.Dense(64, activation='relu'))
# output 64 neurons, avec 1024 * 64 + 64 connections
model.add(layers.Dense(10))
# ouput 10 neurons avec 64 * 10 + 10 connections


model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)


# history = model.fit(train_x, train_y, epochs=30, validation_data=(test_x, test_y))

# https://www.tensorflow.org/tutorials/images/cnn
# https://www.tensorflow.org/tutorials/quickstart/advanced
