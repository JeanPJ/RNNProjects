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
class RNN:

    def __init__(self,neu,n_in,n_out,gama=0.5,ro=1,psi=0.5,in_scale=0.1,bias_scale=0.5,alfa=10,forget = 1-10**-6.0):
        self.Wrr = np.random.normal(0,1,[neu,neu]) #r = reservatorio, i = input, o = output, nomenclatura no modelo Wxy, sendo x a entrada e y a saida. b bias.
        self.Wir = np.random.normal(0,1,[neu,n_in])
        self.Wbr = np.random.normal(0,1,[neu,1])
        self.Wbo = np.random.normal(0,1,[n_out,1])
        self.Wor = np.random.normal(0,1,[n_out,neu])
        self.leakrate = gama
        self.ro = ro
        self.psi = psi
        self.in_scale = in_scale
        self.bias_scale = bias_scale
        self.alfa = alfa
        self.forget = forget
        self.Wrr = Sparcity(self.Wrr,self.psi)
        eigs = np.linalg.eigvals(self.Wrr)
        print type(eigs)
        radius = np.abs(np.max(eigs))
        self.Wrr = self.Wrr/radius
        self.Wrr = self.Wrr*ro


    def Train(self):
        pass
        #rotina de treinamento atraves dos minimos quadrados recursivos atualiza Wor.

    def Update(self):
        pass
        #rotina de atualizacao dos estados, retorna a saida.


RNN(100,1,1,0.9)




