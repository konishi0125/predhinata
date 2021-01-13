import os
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

name_list = pd.read_csv("./member_list.csv")
name_list = name_list["name"].values.tolist()
CATEGORIES = len(name_list)

def load_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    x = data["arr_0"]
    y = data["arr_1"]
    return x, y

def make_model():
    inputs = layers.Input(shape=(512,))
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(CATEGORIES, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=x)
    optimizer = optimizers.Adamax(lr=0.0002)
    model.compile(loss="categorical_crossentropy",
             optimizer=optimizer, metrics=["accuracy"])

    return model 

def learn_model(model, x, y, batch_size=32, epochs=50):
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, random_state=0, train_size=0.8)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,\
        validation_data=(x_test, y_test))
    return model, history

x, y = load_data("./pic_numpy/face_net_pic.npz")
model = make_model()
model, history = learn_model(model, x, y, batch_size=64, epochs=500)
history = pd.DataFrame(history.history)
history.to_csv("./history/history.csv")
model.save("./model/facenet_hinata_model.h5")