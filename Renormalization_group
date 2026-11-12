from scipy.sparse import csr_array
import scipy 
from scipy import linalg
import numpy as np
import math
import torch
import random as rand
import itertools
import matplotlib.pyplot as plt
import cuquantum
print(cuquantum.__version__)
import cupy; cupy.show_config()
print(cupy.__version__)
print(cupy.cuda.runtime.getDeviceCount())
print(cupy.cuda.runtime.runtimeGetVersion())  # e.g. 12040 = CUDA 12.4

#build init tensor
def buildinittensor(beta, J, h):
    Tinit = np.zeros([2,2,2,2])
    spins = [-1,1]
    for r, m, p, k in itertools.product(spins,spins,spins,spins):
        sigmai = r
        sigmaj = m
        sigmak = p
        sigmal = k
        intfactor = beta*J*((sigmai*sigmaj)+(sigmaj*sigmak)+(sigmak*sigmal)+(sigmal*sigmai))
        extfactor = h*beta*(sigmai + sigmaj + sigmak + sigmal)/2
        expfactor = (np.exp(intfactor+extfactor))
        Tinit[max(0,r)][max(0,m)][max(0,p)][max(0,k)] = expfactor
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

    Coursegrainedtensor = cuquantum.tensornet.einsum('abi,caj,dck,bdl->ijkl', uas, ubs, vas, vbs) #AI pointed out a bug here, derivation of the math for this contraction is mine though
    return Coursegrainedtensor, DN, SAVal, SBVal


def iteratemain(Coupling,Temp, nn, eigvalskeep, extfieldvariable):
        J = Coupling
        Trelative = Temp
        Tc = (2*J)/((np.log(1+ np.sqrt(2))))
        Tthing = Trelative*Tc
        beta = 1 / (Tthing)
        normalizationvals = []
        n = nn
        Tensors1 = 2**n
        Eigvalueskeep = eigvalskeep
        DNlist = []
        Tensor1thing = buildinittensor(beta,J, extfieldvariable)
        Tensor1renormalized = Tensor1thing/np.max(Tensor1thing)
        normalizationvals.append(np.max(Tensor1thing))
        Tensorcoursegrained, DN1, SAVal, SBVal = coursegrain(Tensor1renormalized, Eigvalueskeep)
        normalizationvals.append(SAVal*SBVal)
        Tensorcoursegrained = Tensorcoursegrained/(SAVal*SBVal)
        DNlist.append(DN1)
        Tensors = Tensors1/2
        Tensorcoursegrained, DN1, SAVal, SBVal = coursegrain(Tensorcoursegrained, Eigvalueskeep)
        normalizationvals.append(SAVal*SBVal)
        Tensorcoursegrained = Tensorcoursegrained/(SAVal*SBVal)
        DNlist.append(DN1)
        Tensors = Tensors/2
        DN1 = Eigvalueskeep
        while Tensors >= 2:
                Tensorcoursegrained, DN2, SAVal, SBVal = coursegrain(Tensorcoursegrained, DN1)
                normalizationvals.append(SAVal*SBVal)
                Tensorcoursegrained = Tensorcoursegrained/(SAVal*SBVal)
                DNlist.append(DN2)
                DN1 = DN2
                print("lattices:", Tensors)
                print("Temp:",Temp)
                print("Tensor:",Tensorcoursegrained)
                if Tensors % 2 == 0 and Tensors != 0:
                    Tensors = Tensors/2
                else:
                      print("not good")

        print("final tensor:",Tensorcoursegrained)
        print("eigenvalues kept:", DNlist)
        Partfunction1 = np.einsum('ijij->', Tensorcoursegrained) #AI pointed this out, but found it in a paper
        Partfunction = Partfunction1
        print("Partition Function, temp:", Partfunction, Temp)

        #Free energy
        free_energy_weight = 0
        for i in np.arange(len(normalizationvals)):
            free_energy_weight += (1/(2**(i+1)))*np.log(normalizationvals[i])
        A = -Tthing*((2*Tensors1*free_energy_weight)+(np.log(Partfunction)))#-Tthing*math.log(Partfunction) 
        print("free energy", A)
        
        fepus = A/(2*Tensors1) #AI pointed out I had made an error with the total number of sites
        print("free energy per units site:",fepus)
        return fepus, A, Partfunction


#MAIN
eigvalarray = [4,8,16,32,64]
eigvals = np.array(eigvalarray)
temps = np.arange(0.8,1.1,0.01)
errorlist = np.zeros((len(eigvals),len(temps)))
for eigval in np.arange(len(eigvals)):
    couplingvar = 1
    extfieldvar = 0
    numthingsites = 10
    eigvalvar = eigvals[eigval]
    differentialdifferent = 1e-4
    badcounter = 0

    fepuslist = []
    tempslist = []
    coupleslist = []
    fesyslist = []
    partfunclist = []
    fepuslistep = []
    tempslistep = []
    coupleslistep = []
    fesyslistep = []
    partfunclistep = []
    fepuslistem = []
    fesyslistem = []
    partfunclistem = []
    ep = extfieldvar + differentialdifferent
    #em = extfieldvar - differentialdifferent
    #diff = ep - em

    for Temp in temps:
        print("Temperature:",Temp)
        fepustemp, fesystemp, partfunc, = iteratemain(couplingvar, Temp, numthingsites, eigvalvar, extfieldvar)
        fepuslist.append(fepustemp)
        coupleslist.append(couplingvar)
        tempslist.append(Temp)
        fesyslist.append(fesystemp)
        partfunclist.append(partfunc)
        fepustempep, fesystempep, partfuncep, = iteratemain(couplingvar, Temp, numthingsites, eigvalvar, ep)
        fepuslistep.append(fepustempep)
        fesyslistep.append(fesystempep)
        partfunclistep.append(partfuncep)
        #fepustempem, fesystempem, partfuncem, = iteratemain(couplingvar, Temp, numthingsites, eigvalvar, em)
        #fepuslistem.append(fepustempem)
        #fesyslistem.append(fesystempem)
        #partfunclistem.append(partfuncem)


    magnetfin = np.zeros(len(partfunclist))
    for i in np.arange(len(partfunclist)):
        #Using m = - df(h,T)/dh|h->0
        magnetfin[i] = (-((fepuslistep[i] - fepuslist[i])/(differentialdifferent))) #AI linked me to a source for this method of differentiation

    #exact solution (AI helped rewrite with this due to a bug in my function)
    #    K = couplingvar*beta
    Tc = 2*couplingvar/np.log(1+np.sqrt(2))
    theta = np.linspace(0, np.pi, 20000)
    freeenergy = np.zeros(len(temps))
    for plp in np.arange(len(temps)):
        T = temps[plp]*Tc
        K2 = 2*couplingvar/T
        k = 1/np.sinh(K2)**2
        integrand = np.log(np.cosh(K2)**2 + np.sqrt(1 + k**2 - 2*k*np.cos(2*theta))/k)
        freeenergy[plp] = -T*(np.log(2)/2 + np.trapezoid(integrand, theta)/(2*np.pi))

    #Magnetization Exact Solution
    Mexact = np.zeros(len(temps))
    for plp in np.arange(len(temps)):
        if temps[plp]<1:
            Temperature = temps[plp]*Tc
            beta = 1/Temperature
            Mexact[plp] = (((1-(np.sinh(2**couplingvar*beta)**(-4)))**(1/8)))
        else:
            Mexact[plp] = 0
        

    error = []
    np.array(magnetfin)
    np.array(Mexact)
    error = abs((fepuslist - freeenergy)/freeenergy)
    errormag = []
    errormag = abs((magnetfin - Mexact)/Mexact)
    for i in np.arange(len(errormag)):
        errorlist[eigval][i] = errormag[i]
    
    plt.figure(1)
    plt.xlabel("Temperature")
    plt.ylabel("Free energy per unit site (Joules)")
    plt.plot(tempslist,fepuslist, 'x', color = "green")
    plt.plot(tempslist,fepuslistep, 'x', color = "red")
    plt.plot(tempslist,freeenergy, 'x', color = "blue")
    #plt.plot(tempslist,fepuslistem, 'x', color = "orange")

    plt.figure(2)
    plt.xlabel("Temperature (Unitless)")
    plt.ylabel("Error (%)")
    plt.plot(tempslist,error, color = "blue")

    plt.figure(3)
    plt.xlabel("temperature")
    plt.ylabel("magnetization")
    plt.plot(tempslist,magnetfin, 'x', color = "red")
    
plt.figure(4)
plt.xlabel("Temperature")
plt.ylabel("Error In Magnetization (%)")
for i in np.arange(len(errorlist)):
    plt.plot(tempslist,errorlist[i], 'x', label = eigvals[i])
plt.legend()
plt.show()
