from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
# from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##################DISCRIMINATOR##################
def buildDiscriminator():
    inputs = keras.Input(shape=(28,28,1))

    d1 = keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding='same')(inputs)
    d2 = keras.layers.LeakyReLU(0.2)(d1)
    d3 = keras.layers.Dropout(.4)(d2)

    d4 = keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same')(d3)
    d5 = keras.layers.LeakyReLU(0.2)(d4)
    d6 = keras.layers.Dropout(.4)(d5)

    d7 = keras.layers.Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding='same')(d6)
    d8 = keras.layers.LeakyReLU(0.2)(d7)
    d9 = keras.layers.Dropout(.4)(d8)

    d10 = keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding='same')(d9)
    d11 = keras.layers.LeakyReLU(0.2)(d10)
    d12 = keras.layers.Dropout(.4)(d11)

    d13 = keras.layers.Flatten()(d12)
    d14 = keras.layers.Dense(units=1)(d13)
    discriminator = keras.layers.Activation('sigmoid')(d14)
    discriminator = keras.Model(inputs=inputs, outputs=discriminator)
    discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=.0002, beta_1=.5), loss='binary_crossentropy') 
    discriminator.trainable=False
    return discriminator

##################GENERATOR##################
def buildGenerator():
    inputs = keras.Input(shape=[100]) 
    d1 = keras.layers.Dense(units=(7*7*192))(inputs)
    d2 = keras.layers.BatchNormalization()(d1)
    d3 = keras.layers.Activation('relu')(d2)
    d4 = keras.layers.Reshape((7,7,192))(d3)
    d5 = keras.layers.Dropout(.4)(d4)

    # d6 = keras.layers.UpSampling2D(size=(2,2))(d5)
    d6 = keras.layers.UpSampling2D(2)(d5)
    d7 = keras.layers.Conv2DTranspose(filters=96, kernel_size=(5,5), strides=1, padding='same')(d6)
    d8 = keras.layers.BatchNormalization()(d7)
    d9 = keras.layers.Activation('relu')(d8)

    # d10 = keras.layers.UpSampling2D(size=(2,2))(d9)
    d10 = keras.layers.UpSampling2D(2)(d9)
    d11 = keras.layers.Conv2DTranspose(filters=48, kernel_size=(5,5), strides=1, padding='same')(d10)
    d12 = keras.layers.BatchNormalization()(d11)
    d13 = keras.layers.Activation('relu')(d12)

    d14 = keras.layers.Conv2DTranspose(filters=24, kernel_size=(5,5), strides=1, padding='same')(d13)
    d15 = keras.layers.BatchNormalization()(d14)
    d16 = keras.layers.Activation('relu')(d15)

    d17 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(5,5), strides=1, padding='same')(d16)
    generator = keras.layers.Activation('sigmoid')(d17)
    generator = keras.Model(inputs=inputs, outputs=generator)
    generator.compile(optimizer=keras.optimizers.Adam(learning_rate=.0002, beta_1=.5), loss='binary_crossentropy') 
    return generator

##################MAIN##################

# VALUES
epochs = 15
numBatches = 64
quarter = int(epochs / 4);

# Loading data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.astype("float32") / 255

# Building the Models
disModel = buildDiscriminator()
genModel = buildGenerator()
disModel.summary()
genModel.summary()

# Constructing the GAN
GANModel = Sequential()
GANModel.add(genModel)
GANModel.add(disModel)
GANModel.compile(optimizer=keras.optimizers.Adam(learning_rate=.0002, beta_1=.5), loss='binary_crossentropy')
GANModel.summary()

# Initiating the losses.
disLosses=[]
GANLosses=[]
# Intiating correct/loss for discrim.
oneArr = np.ones(numBatches)
zeroArr = np.zeros(numBatches)

for epoch in range(1, epochs+1):
    print("Epoch: "+str(epoch))
    GANLoss = 0
    disLossAvg = 0
    for b in range(numBatches):
        # Gets randomdistribution of values.
        randomVals = np.random.normal(0,1,size=(numBatches, 100))
        imageReal = x_train[np.random.randint(0, x_train.shape[0], size=numBatches)]
        # Allows for it to be the (None, 28, 28, 1). Fixes issue.
        imageReal = imageReal.reshape(imageReal.shape[0], imageReal.shape[1], imageReal.shape[2], 1)
        imageFake = genModel.predict(randomVals)
        
        # Trains the Descriminator with a real value and a fake one.
        disModel.trainable=True
        disLoss1 = disModel.train_on_batch(imageFake, zeroArr)
        disLoss2 = disModel.train_on_batch(imageReal, oneArr)
        disLossAvg = (disLoss1 + disLoss2) / 2
        disModel.trainable=False

        # Trains the GAN with randomVals.
        GANLoss = GANModel.train_on_batch(randomVals, oneArr)
    
    # Printing out the specific EPOCHS for viewing.
    if (epoch == 1 or epoch % quarter == 0 or epoch == epochs): 
        randomVals = np.random.normal(0,1,size=[numBatches,100])
        img = genModel.predict(randomVals)

        plt.figure(figsize=(28,28))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow(img[i], cmap='gray')
        plt.title("Epoch: "+str(epoch))
        plt.show()
    
    disLosses.append(disLossAvg)
    GANLosses.append(GANLoss)

plt.plot(disLosses, label='Loss of Discriminator')
plt.plot(GANLosses, label='Loss of GAN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()