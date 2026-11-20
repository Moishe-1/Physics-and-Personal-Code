from scipy.sparse import csr_array
import scipy as scp
from scipy import linalg
import numpy as np
import math
import torch
import random as rand
import numpy.linalg as linalg
import itertools
import matplotlib.pyplot as plt
import cuquantum
print(cuquantum.__version__)
import cupy; cupy.show_config()
print(cupy.__version__)
print(cupy.cuda.runtime.getDeviceCount())
print(cupy.cuda.runtime.runtimeGetVersion())  # e.g. 12040 = CUDA 12.4

#build init tensor
def buildinittensor(beta, J):
    Tinit = np.zeros([2,2,2,2])
    spins = [-1,1]
    print(Tinit)
    for r, m, p, k in itertools.product(spins,spins,spins,spins): #Pointed out error in change of variables
        sigmamult = (r)*(m)*(p)*(k)
        #print(sigmai+sigmaj+sigmak+sigmal)
        intfactor = beta*J*((r)+(m)+(p)+(k))/2
        expfactor = (np.exp(intfactor))
        sigmamult = 0.5*(sigmamult+1)
        Tinit[max(0,r)][max(0,m)][max(0,p)][max(0,k)] = sigmamult*expfactor
    return(Tinit)

#Course Grain Operation
def coursegrain(Tensor1, eigvalsnum):
    
    #Tensor A breakup
    ua, sa, va = cuquantum.tensornet.tensor.decompose('ruld->ulm,drm', Tensor1, method=cuquantum.tensornet.tensor.SVDMethod(max_extent=DN))#,partition="UV"
    print("UA shape:",ua.shape[2])
    print("VA shape:",va.shape[0])
    SAVal = np.max(sa)
    sqrta = np.sqrt(sa)
    uas = np.einsum('ulm,m->ulm',ua,sqrta)
    vas = np.einsum('drm,m->drm',va,sqrta)

    #Tensor B breakup
    ub, sb, vb = cuquantum.tensornet.tensor.decompose('ruld->ldm,rum', Tensor1, method=cuquantum.tensornet.tensor.SVDMethod(max_extent=DN))#,partition="UV"
    SBVal = np.max(sb)
    sqrtb = np.sqrt(sb)
    ubs = np.einsum('ldm,m->ldm',ub,sqrtb)
    vbs = np.einsum('rum,m->rum',vb,sqrtb)

    Coursegrainedtensor = cuquantum.tensornet.einsum('abi,caj,dck,bdl->ijkl', uas, ubs, vas, vbs)
    return Coursegrainedtensor, DN, SAVal, SBVal


def iteratemain(Coupling,Temp, nn, eigvalskeep):
        J = Coupling
        Tc = (2*Temp)/((np.log(1+ np.sqrt(2))))
        Tthing = Tc
        beta = 1 / (Tthing)
        normalizationvals = []
        Tensors1 = 2**nn
        Eigvalueskeep = eigvalskeep
        DNlist = []
        Tensor1thing = buildinittensor(beta,J)
        Tensor1renormalized = Tensor1thing/np.max(Tensor1thing)
        normalizationvals.append(np.max(Tensor1thing))
        Tensorcoursegrained, DN1, SAVal, SBVal = coursegrain(Tensor1renormalized, Eigvalueskeep)
        Tensorcoursegrained = Tensorcoursegrained/(SAVal*SBVal)
        normalizationvals.append(SAVal*SBVal)
        DNlist.append(DN1)
        Tensors = Tensors1/2
        DN1 = Eigvalueskeep
        while Tensors >= 2:
                Tensorcoursegrained, DN2, SAVal, SBVal = coursegrain(Tensorcoursegrained, DN1)
                Tensorcoursegrained = Tensorcoursegrained/(SAVal*SBVal)
                normalizationvals.append(SAVal*SBVal)
                DNlist.append(DN2)
                DN1 = DN2
                print("lattices:", Tensors)
                if Tensors % 2 == 0 and Tensors != 0:
                    Tensors = Tensors/2
                else:
                      print("not good")

        Partfunction = np.einsum('ijij->', Tensorcoursegrained)

        #Free energy
        free_energy_weight = 0
        for i in np.arange(len(normalizationvals)):
            free_energy_weight += (1/(2**(i+1)))*np.log(normalizationvals[i])
        A = (2*Tensors1*free_energy_weight)
        print("free energy", A)
        fepus = A/(2*Tensors1)
        print("K =", Coupling, " lnZ/N =", math.log(Partfunction)/Tensors1, " f =", fepus)
        return fepus, A, Partfunction


#MAIN  
temp = 1                    
couplingvar = np.arange(0.01,1.7,0.05)
numthingsites = 5
eigvalvar = 16

fepuslist = []
tempslist = []
coupleslist = []
fesyslist = []
partfunclist = []
couplinglist = []

for Couple in couplingvar:
    print("Coupling:",Couple)
    fepustemp, fesystemp, partfunc, = iteratemain(Couple, temp, numthingsites, eigvalvar)
    fepuslist.append(fepustemp)
    tempslist.append(temp)
    fesyslist.append(fesystemp)
    partfunclist.append(partfunc)

#exact solution
theta = np.linspace(0, np.pi, 20000)
freeenergy = np.zeros(len(couplingvar))
for plp in np.arange(len(couplingvar)):
    Tc = temp
    K2 = 2*couplingvar[plp]/Tc         
    k = 1/np.sinh(K2)**2
    integrand = np.log(np.cosh(K2)**2 + np.sqrt(1 + k**2 - 2*k*np.cos(2*theta))/k)
    freeenergy[plp] = (np.log(2)/2 + np.trapezoid(integrand, theta)/(2*np.pi))

#plotting
plt.figure(2)
plt.title("Free Energy Per Unit Site against coupling variable")
plt.xlabel("coupling variable")
plt.ylabel("free energy")
plt.plot(couplingvar,fepuslist, 'x', color = "green",label='Free energy Computed')
plt.plot(couplingvar,freeenergy, 'x', color = "orange", label='Exact solution')
plt.axvline(x = 0.441, color = 'b', label = 'Critical Temperature')
plt.legend()
plt.show()
