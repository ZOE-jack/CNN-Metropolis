import numpy as np

# Create the input data with label
data1 = np.load("./data/input1.npy", allow_pickle=True)
data1 = data1.reshape(80*1000, 64, 64)
data1_train_target = np.array([i for i in range(0,16) for _ in range(5000)])
with open("./data/training/train_data.npy", "wb") as outfile:
    np.save(outfile, {"inputs": data1, "targets": data1_train_target})

