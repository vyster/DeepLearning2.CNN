# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()


classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(227, 227, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

classifier.add(Convolution2D(256, 5, strides = (1, 1), padding='same', activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))

classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))

classifier.add(Convolution2D(256, 3, strides=(1,1), padding='same', activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))

classifier.add(Dense(units = 38, activation = 'softmax'))

classifier.summary()
