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
class RNN

    def __init__(self,neu,n_in,n_out,gama,ro,psi,in_scale,bias_scale,alfa,forget = 1-10**-6.0,reg):
        self.Wrr = np.random.normal(0,1,[neu,neu]) #r = reservatorio, i = input, o = output, nomenclatura no modelo Wxy, sendo x a entrada e y a saida. b bias.
        self.Wir = np.random.normal(0,1,[neu,n_in])
        self.Wbr = np.random.normal(0,1,[neu,1])
        self.Wbo = np.random.normal(0,1,[n_out,1])
        self.leakrate = gama
        self.ro = ro
        self.psi = psi
        self.in_scale = in_scale
        self.bias_scale = bias_scale
        self.alfa = alfa
        self.forget = forget
        self.reg = reg
        self.Wrr = Sparcity(self.Wrr,self.psi)

    def Train:
        #rotina de treinamento atraves dos minimos quadrados recursivos.

    def update:
        #rotina de atualizacao dos estados, retorna a saida.




