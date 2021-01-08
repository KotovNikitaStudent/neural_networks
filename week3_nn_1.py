# import numpy as np
# import tensorflow as tf
#
# # 2D convolution layer (e.g. spatial convolution over images).
# tf.keras.layers.Conv2D(
#     filters, kernel_size, strides=(1, 1), padding='valid',
#     data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
#     use_bias=True, kernel_initializer='glorot_uniform',
#     bias_initializer='zeros', kernel_regularizer=None,
#     bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
#     bias_constraint=None, **kwargs)
#
# # This layer creates a convolution kernel that is convolved
# # with the layer input to produce a tensor of outputs.
# # If use_bias is True, a bias vector is created and added to the outputs.
# # Finally, if activation is not None, it is applied to the outputs as well.
# #
# # When using this layer as the first layer in a model,
# # provide the keyword argument input_shape
# # (tuple of integers, does not include the sample axis),
# # e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in
# # data_format="channels_last".
#
# # The inputs are 28x28 RGB images with `channels_last` and the batch
# # size is 4.
# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(
# 2, 3, activation='relu', input_shape=input_shape[1:])(x)
# print(y.shape)
#
# # With `dilation_rate` as 2.
# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(
# 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
# print(y.shape)
#
# # With `padding` as "same".
# input_shape = (4, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(
# 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
# print(y.shape)
#
# # With extended batch shape [4, 7]:
# input_shape = (4, 7, 28, 28, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(
# 2, 3, activation='relu', input_shape=input_shape[2:])(x)
# print(y.shape)
#
# # Max pooling operation for 2D spatial data.
#
# # Downsamples the input representation by taking the maximum
# # value over the window defined by pool_size for
# #   each dimension along the features axis.
# #   The window is shifted by strides in each dimension.
# #   The resulting output when using "valid" padding option
# #   has a shape(number of rows or columns)
# #   of: output_shape = (input_shape - pool_size + 1) / strides)
# #
# # The resulting output shape when using the "same"
# #   padding option is: output_shape = input_shape / strides
# #
# # For example, for stride=(1,1) and padding="valid":
#
# tf.keras.layers.MaxPool2D(
#     pool_size=(2, 2), strides=None, padding='valid', data_format=None,
#     **kwargs
# )
# #For example, for stride=(1,1) and padding="valid":
# x = tf.constant([[1., 2., 3.],
#                  [4., 5., 6.],
#                  [7., 8., 9.]])
# x = tf.reshape(x, [1, 3, 3, 1])
# max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
#    strides=(1, 1), padding='valid')
# max_pool_2d(x)
#
# # For example, for stride=(2,2) and padding="valid":
#
# x = tf.constant([[1., 2., 3., 4.],
#                  [5., 6., 7., 8.],
#                  [9., 10., 11., 12.]])
# x = tf.reshape(x, [1, 3, 4, 1])
# max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
#    strides=(1, 1), padding='valid')
# max_pool_2d(x)
#
#
# input_image = tf.constant([[[[1.], [1.], [2.], [4.]],
#                            [[2.], [2.], [3.], [2.]],
#                            [[4.], [1.], [1.], [1.]],
#                            [[2.], [2.], [1.], [4.]]]])
# output = tf.constant([[[[1], [0]],
#                       [[0], [1]]]])
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
#    input_shape=(4,4,1)))
# model.compile('adam', 'mean_squared_error')
# model.predict(input_image, steps=1)
#
#
# x = tf.constant([[1., 2., 3.],
#                  [4., 5., 6.],
#                  [7., 8., 9.]])
# x = tf.reshape(x, [1, 3, 3, 1])
# max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
#    strides=(1, 1), padding='same')
# max_pool_2d(x)



# import tensorflow as tf
# # print(tf.__version__)
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# training_images=training_images.reshape(60000, 28, 28, 1)
# training_images=training_images / 255.0
# test_images = test_images.reshape(10000, 28, 28, 1)
# test_images=test_images/255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2,2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(training_images, training_labels, epochs=5)
# # test_loss = model.evaluate(test_images, test_labels)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(test_acc)
#
# import matplotlib.pyplot as plt
# f, axarr = plt.subplots(3,4)
# FIRST_IMAGE=0
# SECOND_IMAGE=7
# THIRD_IMAGE=26
# CONVOLUTION_NUMBER = 1
# from tensorflow.keras import models
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
# for x in range(0,4):
#   f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[0,x].grid(False)
#   f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[1,x].grid(False)
#   f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
#   axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
#   axarr[2,x].grid(False)


import tensorflow as tf
# print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
def train_model():
  class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if (logs.get('accuracy') >= 0.99):
        print('\nReached 99% accuracy so cancelling training!')
        self.model.stop_training = True
  callbacks = myCallback()

  history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
  return history.epoch, history.history['accuracy'][-1]
train_model()
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)

# import cv2
# import numpy as np
# from scipy import misc
# i = misc.ascent()
#
# import matplotlib.pyplot as plt
# plt.grid(False)
# plt.gray()
# plt.axis('off')
# plt.imshow(i)
# plt.show()
#
# i_transformed = np.copy(i)
# size_x = i_transformed.shape[0]
# size_y = i_transformed.shape[1]
#
# # This filter detects edges nicely
# # It creates a convolution that only passes through sharp edges and straight
# # lines.
#
# #Experiment with different values for fun effects.
# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
#
# # A couple more filters to try for fun!
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# #filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#
# # If all the digits in the filter don't add up to 0 or 1, you
# # should probably do a weight to get it to do so
# # so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# # They add up to 10, so you would set a weight of .1 if you want to normalize them
# weight  = 1
#
# for x in range(1,size_x-1):
#   for y in range(1,size_y-1):
#       convolution = 0.0
#       convolution = convolution + (i[x - 1, y-1] * filter[0][0])
#       convolution = convolution + (i[x, y-1] * filter[0][1])
#       convolution = convolution + (i[x + 1, y-1] * filter[0][2])
#       convolution = convolution + (i[x-1, y] * filter[1][0])
#       convolution = convolution + (i[x, y] * filter[1][1])
#       convolution = convolution + (i[x+1, y] * filter[1][2])
#       convolution = convolution + (i[x-1, y+1] * filter[2][0])
#       convolution = convolution + (i[x, y+1] * filter[2][1])
#       convolution = convolution + (i[x+1, y+1] * filter[2][2])
#       convolution = convolution * weight
#       if(convolution<0):
#         convolution=0
#       if(convolution>255):
#         convolution=255
#       i_transformed[x, y] = convolution
#
# # Plot the image. Note the size of the axes -- they are 512 by 512
# plt.gray()
# plt.grid(False)
# plt.imshow(i_transformed)
# #plt.axis('off')
# plt.show()
#
# new_x = int(size_x/2)
# new_y = int(size_y/2)
# newImage = np.zeros((new_x, new_y))
# for x in range(0, size_x, 2):
#   for y in range(0, size_y, 2):
#     pixels = []
#     pixels.append(i_transformed[x, y])
#     pixels.append(i_transformed[x+1, y])
#     pixels.append(i_transformed[x, y+1])
#     pixels.append(i_transformed[x+1, y+1])
#     newImage[int(x/2),int(y/2)] = max(pixels)
#
# # Plot the image. Note the size of the axes -- now 256 pixels instead of 512
# plt.gray()
# plt.grid(False)
# plt.imshow(newImage)
# #plt.axis('off')
# plt.show()

# Использование возможностей gpu
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)