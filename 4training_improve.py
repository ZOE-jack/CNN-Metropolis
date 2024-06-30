import tensorflow as tf
import keras
import numpy as np
from sklearn.utils import shuffle 
import time
import json
import matplotlib.pyplot as plt

# import the model
model = keras.models.load_model("./data/model/model_regular.keras")

# prepare data
x_data = np.load("./data/training/train_data.npy", allow_pickle=True).item()["inputs"]
y_data = np.load("./data/training/train_data.npy", allow_pickle=True).item()["targets"]

# shuffle the datasets
x_data, y_data = shuffle(x_data, y_data)

# Normalized the datasets to [0,1]
x_data = (x_data+1)/2

# Take the first 50k datasets 
x_data = x_data[:50000]
y_data = y_data[:50000]

# Split them into 1:4 for test and train datasets
x_train, x_test = x_data[:10000], x_data[10000:50000]
y_train, y_test = y_data[:10000], y_data[10000:50000]

# methods
OPTIMIZER = keras.optimizers.Adam(learning_rate = 0.001)
LOSS = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
METRICS = ['accuracy']
model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics=METRICS)

# save the best parameter
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./data/chk_points/best.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# training 
EPOCHS = 30
BATCH_SIZE = 50
VALIDATION = (x_test, y_test)
since = time.time()
history = model.fit(x_train,
                    y_train,
                    batch_size = BATCH_SIZE, 
                    epochs = EPOCHS,
                    validation_data = VALIDATION,
                    callbacks = [model_checkpoint_callback])
print(f"It takes {(time.time()-since)/60:.4f} to train the model.")

# Plot the loss curve
plt.plot(history.history["loss"], "o-",label="Train_loss")
plt.plot(history.history["val_loss"] , "o-", label="Val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss curve")
plt.grid()

# Plot the accuracy
plt.plot(history.history["accuracy"], "o-",label="Train_accuracy")
plt.plot(history.history["val_accuracy"] , "o-", label="Val_accuracy")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("Loss curve")
plt.grid()