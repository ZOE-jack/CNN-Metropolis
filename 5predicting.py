import tensorflow as tf
import keras
import numpy as np
import time
import json
import matplotlib.pyplot as plt

# import the model
model = keras.models.load_model("./data/model/model_regular.keras")

# load the best weight of the model
model.load_weights("./data/chk_points/best_regular.h5")

# prepare data
x_data = np.load("./data/training/train_data.npy", allow_pickle=True).item()["inputs"]
y_data = np.load("./data/training/train_data.npy", allow_pickle=True).item()["targets"]

# Do normalize to make data between [0,1]
x_data = (x_data+1)/2

# Predict the data with certain temperature label
record = []
for i in range(16):
    m = keras.metrics.SparseCategoricalAccuracy()
    m.update_state(y_data[i*(5000):(i+1)*(5000)], model.predict(x_data[i*(5000):(i+1)*(5000)]))
    record.append(m.result())

# Plot the graph
plt.plot(np.arange(1.5,3.01, 0.1), record, "-o")
plt.title("Accuracy per Temperature")
plt.xlabel("Temparature")
plt.ylabel("Accuracy")
plt.grid()