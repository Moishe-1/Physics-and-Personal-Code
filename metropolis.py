import random
from numba import njit, prange
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage
import math

Kb =1#1.38*(10**-23)
def buildlattice(length,width):
    lattice = np.zeros((length,width))
    for i in np.arange(length):
        for j in np.arange(width):
            lattice[i][j] = random.choice([-1, 1])
    return lattice

@njit(cache=True, fastmath=True)
def magnetization(lattice,length,width):
    magnetization = abs((np.sum(lattice))/(length*width))
    return magnetization

@njit(cache=True, fastmath=True)
def nearest_neighbors_ith(i,j,lattice,jfield):
    n,m = lattice.shape
    
    leftvalue = jfield*lattice[(i-1)%n][(j)]*lattice[i][j]
    rightvalue = jfield**lattice[(i+1)%n][j]*lattice[i][j]
    upvalue = jfield*lattice[(i)][(j-1)%m]*lattice[i][j]
    downvalue = jfield*lattice[(i)][(j+1)%m]*lattice[i][j]
    internalfieldeff = -(downvalue+upvalue+rightvalue+leftvalue)
    
    return internalfieldeff

@njit(cache=True, fastmath=True)
def nearest_neighbors(i,j,lattice,jfield,hfield):
    l = 1
    internalfieldtot = 0
    while l <= m:
        internalfieldtot += nearest_neighbors_ith(i,j,lattice,jfield)
        l += 1
    externalfieldeff = -hfield*lattice[i][j]
    fieldeff = internalfieldtot + externalfieldeff
    return fieldeff

@njit(cache=True, fastmath=True)
def flipalgorithm(lattice, length, width, jfield, hfield, beta):
    choice1 = np.random.randint(length)
    choice2 = np.random.randint(width)
    fieldeffinit = nearest_neighbors(choice1,choice2,lattice,jfield,hfield)
    lattice2 = lattice.copy()
    lattice2[choice1][choice2] = -lattice2[choice1][choice2]
    fieldeffafter = nearest_neighbors(choice1,choice2,lattice2,jfield,hfield)
    #print(beta)
    if fieldeffafter - fieldeffinit > 0: #AI corrected a couple errors in here
        x = np.exp(-beta*(fieldeffafter-fieldeffinit))
        #print(x)
        #print(fieldeffafter-fieldeffinit, x)
        if np.random.rand() <= x:
            #print("Found!")
            return lattice2
        else:
            return lattice
    else:
        return lattice2
    
@njit(cache=True, fastmath=True) #Ai suggested this form of iteration for parallel computing
def sweep(lattice2,length,width,jfield,hfield,beta):
    g = 0
    while g < (length*width):
        lattice2 = flipalgorithm(lattice2, length, width, jfield, hfield, beta)
        g += 1
    return lattice2


jfield = 1
hfield = 0
alpha = 1.2
Tc = (2*jfield)/(Kb*(np.log(1+ np.sqrt(2))))
temps1 = np.arange(0.4,1.2,0.01)
Temps = temps1*Tc
magnetizationlist = []
j = []
Tempslist = []
tsteps = (10**3)
distance = 1
print("running")
N = [20,30,40,50]
for Nlen in N:
    j = []
    length = Nlen
    width = Nlen
    for T in Temps:
        print(T)
        lattice2 = buildlattice(length,width)
        beta = 1/(Kb*T)
        m = 0
        while m < tsteps:
            lattice2 = sweep(lattice2,length,width,jfield,hfield,beta)
            magnetizationlist.append((magnetization(lattice2,length,width)))
            m += 1
        j.append((sum(magnetizationlist)/(length*width*tsteps)))
        


    #Tempslist.append(0.7)
    #for i in np.arange(len(temps1)):
    #    Tempslist.append(temps1[i])
    #plt.figure(1)
    #ani = animation.ArtistAnimation(fig, ims, interval = 2, blit = True, repeat_delay = 100)

    plt.figure(2)
    plt.title("Permanent Magnet Phase Transition")
    plt.xlabel("Temperature Percentage")
    plt.ylabel("Magnetization")
    plt.plot(temps1,j, 'x', color = "red")
    plt.show()
