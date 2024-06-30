import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sizes', metavar='', action="append", type=int)
parser.add_argument('-min_time', metavar='', type=float, default=0.5)
parser.add_argument('-max_time', metavar='', type=float, default=5.0)
parser.add_argument('-time_gap', metavar='', type=float, default=0.1)
parser.add_argument('-burning', metavar='', type=int, default=20000)
parser.add_argument('-recording', metavar='', type=int, default=10000)
parser.add_argument('-rcgap', metavar='', type=int, default=5)
parser.add_argument('-bins', metavar='', type=int, default=1)
parser.add_argument('-svfile', metavar='', type=str, default="none")
opt = parser.parse_args()

@nb.jit
def energy(lattice):
    shape = lattice.shape[0]-1
    energy = 0
    for i in range(shape):
        for j in range(shape):
            if i == shape:
                energy += lattice[i,j] * lattice[1,j]
            else:
                energy += lattice[i,j] * lattice[i+1,j]
            if j == shape:
                energy += lattice[i,j] * lattice[i,1]
            else:
                energy += lattice[i,j] * lattice[i,j+1]
    return energy

@nb.jit
def surround_E(lattice, rowcol, shape):
    i, j = rowcol
    return -lattice[i,j] * (lattice[(i+1)%shape,j] + lattice[i-1,j] + lattice[i,(j+1)%shape] + lattice[i, j-1])

@nb.jit
def create_lattice(size):
    '''Create an initial lattice with random [-1,1] spin'''
    lattice = np.random.choice(np.array([-1,1]), (size,size))
    return lattice, -energy(lattice), np.sum(lattice)

@nb.jit
def boltzman(energy, T, k=1):
    return np.exp(-energy/(k*T))

@nb.jit
def A_metropolis_step(lattice, E, M, walkers, shape, temp):
    point1 = walkers[0]
    for number, point2 in enumerate(walkers[1:]):
        del_E = -2 * surround_E(lattice, point2, shape) # del_E = Ec' - Ec = (-sisj-sisj) * (surrounding spin) #??2
        x2, y2 = point2
        if del_E <= 0:
            lattice[x2, y2] = -lattice[x2, y2]
            E += del_E
            M += 2*(lattice[x2, y2])
        elif np.random.rand(1) < boltzman(del_E, T=temp):
            lattice[x2, y2] = -lattice[x2, y2]
            E = E + del_E
            M = M + 2*(lattice[x2, y2])
    return lattice, E, M

@nb.jit
def single_matrix_metropolis(size, temp, burn_in_mcstep, mcstep, mc_gap, bins):
    M_expval = np.empty((0))
    C = np.empty((0))
    X = np.empty((0))
    U = np.empty((0))
    Lts = np.empty((0, int(mcstep/mc_gap), size, size))
    for t in temp:
        for _ in range(bins):
            E_record = np.empty((0))
            M_record = np.empty((0))
            lattice_record = np.empty((0,size,size))
            lattice, E, M = create_lattice(size)
            arbitrary_walk = np.random.randint(0, size, (burn_in_mcstep, size**2, 2))
            for walk in arbitrary_walk:
                lattice, E, M = A_metropolis_step(lattice, E, M, walk, size, t)
            arbitrary_walk = np.random.randint(0, size, (mcstep, size**2, 2))
            for index, walk in enumerate(arbitrary_walk):
                lattice, E, M = A_metropolis_step(lattice, E, M, walk, size, t)
                E_record = np.append(E_record, E)
                M_record = np.append(M_record, np.abs(M))
                if (index+1)%mc_gap == 0:
                    lattice_record = np.append(lattice_record, lattice.reshape(1,size,size), axis=0)
            M_expval = np.append(M_expval, np.sum(M_record)/len(M_record))
            C = np.append(C, (np.sum(E_record**2)/len(E_record)-(np.sum(E_record)/len(E_record))**2) / t**2)
            X = np.append(X, (np.sum(M_record**2)/len(M_record)-(np.sum(M_record)/len(M_record))**2) / t)
            U = np.append(U, 1 - (np.sum(M_record**4)/len(M_record)) / (3*(np.sum(M_record**2)/len(M_record))**2))
            Lts = np.append(Lts, lattice_record.reshape(1,len(lattice_record),size,size), axis=0)
    return Lts, M_expval/size**2, C/size**2, X/size**2, U

def start(sizes, T, burn_in_mcstep, mcstep, mc_gap, bins):
    print("Notice: A montecarlo step will increase when size go up.",
           "\nIn this condition, you take total", (burn_in_mcstep+mcstep)*(sum(sizes))/10e2,
           "million walkers to run.")
    Ls = []
    Ms = np.empty((0,len(T)*bins))
    Cs = np.empty((0,len(T)*bins))
    Xs = np.empty((0,len(T)*bins))
    Us = np.empty((0,len(T)*bins))
    for size in sizes:
        since = time.time()
        L, M, C, X, U = single_matrix_metropolis(size, T, burn_in_mcstep, mcstep, mc_gap, bins)
        Ls.append(L)
        Ms = np.append(Ms, M.reshape(1,len(T)*bins), axis=0)
        Cs = np.append(Cs, C.reshape(1,len(T)*bins), axis=0)
        Xs = np.append(Xs, X.reshape(1,len(T)*bins), axis=0)
        Us = np.append(Us, U.reshape(1,len(T)*bins), axis=0)
        print("Now we complete the calculation of size", size,".")
        print(f"It takes {(time.time()-since)/60:.4f} mimutes.")
    return Ls, Ms, Cs, Xs, Us

if __name__ == "__main__":
    Temperature = np.arange(opt.min_time, opt.max_time+0.0001, opt.time_gap)
    all_size = np.array(opt.sizes)
    L, M, C, X, U = start(sizes=all_size,
                          T=Temperature,
                          burn_in_mcstep=opt.burning,
                          mcstep=opt.recording,
                          mc_gap=opt.rcgap,
                          bins = opt.bins)
    print("Saving.........")
    with open(f"{opt.svfile}.npy", "wb") as outfile:
        np.save(outfile, {"L": L, "M": M, "C": C, "X": X, "U":U})
    print("complete!")