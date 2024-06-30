import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import animation

# Plot for the figure 2.2
data = np.load("./data/good_16_32_64_tg01.npy", allow_pickle=True).item()
L, M, C, X, U = data["L"], data["M"], data["C"], data["X"], data["U"]
Temperature = np.arange(0.5, 5.1, 0.1)
all_size = np.array([16, 32, 64])
fig, ax = plt.subplots(1, 4, figsize=(23, 5))
labels = ["$16^2$", "$32^2$", "$64^2$"]
for index, data in enumerate([[M,r"$\langle|M|\rangle$"], [C,r"$\sigma^2(E)/T^2$"], [X,r"$\sigma^2(M)/T^2$"], [U,r"$U_L$"]]):
    max_value = np.max(data[0])
    min_value = np.min(data[0])
    ax[index].plot(Temperature, data[0].T, "-o", label=labels[:len(all_size)])
    ax[index].vlines(2.27, min_value, max_value, color="red", linestyle="--", linewidth=3)
    ax[index].set_title(data[1]+" with Temperature")
    ax[index].set_ylabel(data[1])
    ax[index].set_xlabel("Temperature")
    ax[index].grid()
    ax[index].legend(title="M size")
plt.show()

# Plot the Figure 1
Temperature = np.arange(0.5, 5.1, 0.1)
mapping = ListedColormap(["lightskyblue", "navajowhite"])
fig, ax = plt.subplots(1,1)
def init():
    pass
def plotting_func(index):
    figure = ax.pcolormesh(L[0][index][0],cmap=mapping, edgecolors='black', linewidth=0.5, vmin=-1, vmax=1)
    ax.set_title(f"Temp = {0.5 +index*0.1:.1f}")
    if index == 0:
        cbr = fig.colorbar(figure, ticks=[-1,1])
        #cbr.set_ticks()
    plt.gca()
    ax.set_aspect('equal')
animat = animation.FuncAnimation(fig=fig, func=plotting_func, frames=len(Temperature), init_func=init)
writer = animation.PillowWriter(fps=2)
#animat.save('animat_size16.gif', writer=writer)

# Plot the demonstration1 in class

record_all = np.empty((0,46))
for size in range(3):
    record = np.array([])
    for t in range(46):
        record = np.append(record, np.sum(L[size][t])/((16*(2**(size)))**2)/len(L[size][t]))
    record_all = np.append(record_all, record.reshape(1, 46), axis=0)

Temperature = np.arange(0.5, 5.1, 0.1)
labels = ["$16^2$", "$32^2$", "$64^2$"]
plt.plot(Temperature, record_all.T, "-o", label=labels)
plt.vlines(2.27, -1, 1, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Temperature")
plt.ylabel(r"$\langle M \rangle$", rotation=0)
plt.title(r"$\langle M \rangle$ for each Temperature")
plt.grid()
plt.legend()
plt.show()

# Plot the demonstration2 in class
Temperature = np.arange(1.9, 2.51, 0.1)
labels = ["$16^2$", "$32^2$", "$64^2$"]
plt.plot(Temperature, record_all[:,14:21].T, "-o", label=labels)
plt.vlines(2.27, -1, 1, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Temperature")
plt.ylabel(r"$\langle M \rangle$", rotation=0)
plt.title(r"$\langle M \rangle$ for each Temperature")
plt.grid()
plt.legend()