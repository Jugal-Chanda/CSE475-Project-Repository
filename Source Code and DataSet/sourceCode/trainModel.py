import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


trainData = ImageDataGenerator(rescale=1./255)
testData = ImageDataGenerator(rescale=1./255)

trainDataGenerator = trainData.flow_from_directory(
        '../dataset/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

testDataGenerator = testData.flow_from_directory(
        '../dataset/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))

loss = keras.losses.categorical_crossentropy
optimizer = keras.optimizers.Adam(lr=0.001)

model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy']
)

model.fit(
        trainDataGenerator,
        steps_per_epoch=448,
        epochs=30,
        verbose=1,
        validation_data=testDataGenerator,
        validation_steps=112
)
model.save('model.h5')