import re

from keras import metrics
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Dropout, LeakyReLU, UpSampling1D
import time
import numpy as np
from random import randint
#from pymongo import MongoClient
#import  matplotlib.pyplot as plt

epochs = 100
step_per_epoch = 500
samples_per_step = 100 #batch_size
nbEcgBeat = 199
inputShape = (nbEcgBeat, 1)

window_size = 200

data_size = 1250

train_patient = ["U30", "U31", "U32", "U34", "U52", "U36", "U41", "U45", "U46", "U49", "U51", "U35"]
#train_patient = ["U31"]
def getData(ecg_file_name, ann_file_name):
    with open(ecg_file_name, "rb") as f:
        f = f.read()
        sdata = f.decode().split()
        arange = int(round(len(sdata) / 3))
        data = [sdata[i * 3 + 1] for i in range(arange)]
        annMap = dict()
        with open(ann_file_name, "rb") as fa:
            fa = fa.read()
            adata = fa.decode().split('\n')
            for i in range(len(adata)):
                match = re.search("[[](.*)[]][ ]+([0-9]*)[ ]+(.)", adata[i])
                if match != None:
                    atime, idx, ann = match.groups()
                    if ann == 'N':
                        annMap[int(idx)] = 1
        return data, annMap



data, annMap = getData("31.txt", "31_a.txt")

def z_norm(result):
    result_mean = result.mean()
    result_std = result.std()
    result -= result_mean
    result /= result_std
    return result, result_mean

def genTrainData(fold):
    while 1:
        ifrom = fold*1000000
        ito = (fold+1)*1000000 - 1250
        input = []
        output = []
        for i in range(samples_per_step):
            index = randint(0, 5000000)
            while ito > index > ifrom:
                index = randint(0, 5000000)
            raw_input = data[index:index + 200]
            ecg = np.asarray(raw_input, dtype=np.float32)
            ecg, mean = z_norm(ecg)
            input.append(ecg[:-1])
            output.append(ecg[-1])

        input = np.reshape(input,(len(input), 199, 1))
        output = np.asarray(output, dtype=np.float32)

        yield (input, output)
def genTestData(fold):
    while 1:
        ifrom = fold*1000000
        ito = (fold+1)*1000000 - 1250
        input = []
        output = []
        for i in range(samples_per_step):
            index = randint(ifrom, ito)

            raw_input = data[index:index + 200]
            ecg = np.asarray(raw_input,dtype=np.float32)
            ecg, mean = z_norm(ecg)
            input.append(ecg[:-1])
            output.append(ecg[-1])

        input = np.reshape(input,(len(input), 199, 1))
        output = np.asarray(output, dtype=np.float32)

        yield (input, output)
def build_model():
    model = Sequential()
    model.add(UpSampling1D(3, input_shape=(nbEcgBeat, 1)))
    model.add(Conv1D(
        filters=3,
        kernel_size=102
    ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(
        filters=10,
        kernel_size=24
    ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(
        filters=10,
        kernel_size=11
    ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(
        filters=10,
        kernel_size=9
    ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(units=30))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(units=10))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(units=1))
    model.add(Activation("linear"))
    # sgd = optimizers.SGD(lr=0.7, momentum=0.0003)
    start = time.time()

    model.compile(loss="mse", optimizer="rmsprop",
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    #print(model.summary())
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2():

    model = Sequential()
    model.add(Conv1D(
        filters=3,
        kernel_size=21,
        input_shape=inputShape
    ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(
        filters=10,
        kernel_size=11
    ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(
        filters=10,
        kernel_size=5
    ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    model.add(Flatten())
    model.add(Dense(units=20))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(units=10))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(units=1))
    model.add(Activation("linear"))
    #sgd = optimizers.SGD(lr=0.7, momentum=0.0003)
    start = time.time()

    model.compile(loss="mse", optimizer="rmsprop",
                  metrics=[metrics.mae, metrics.categorical_accuracy])
    print(model.summary())
    print("Compilation Time : ", time.time() - start)
    return model


def run(model = None):
    if model is None:
        model = build_model()
    print(model.summary())


    model = build_model()
    filepath = "model.{epoch:02d}.h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1)
    model.fit_generator(genTrainData(), epochs=epochs, steps_per_epoch=step_per_epoch,
                        callbacks=[checkpointer], validation_data=genTestData(), validation_steps=50,
                        verbose=2)


#run()
