__author__ = 'jean'
import numpy as np

def Sparcity(M,psi):
    N = np.empty_like(M)
    for linha in range(len(N)):
        for coluna in range(len(N[linha])):
            prob = np.random.rand()
            if prob < psi:
                N[linha][coluna] = 0
            else:
                N[linha][coluna] = 1

    return N*M





#Oi = np.empty([4,4])



#Oi = Sparcity(Oi,0.5)


#print Oi