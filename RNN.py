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
        #All matrixes are initialized under the normal distribution.

        self.neu = neu
        self.n_in = n_in
        self.n_out = n_out

        self.Wrr = np.random.normal(0,1,[neu,neu]) #Reservoir Weight matrix.
        self.Wir = np.random.normal(0,1,[neu,n_in]) #input-reservoir weight matrix
        self.Wbr = np.random.normal(0,1,[neu,1])    #bias-reservoir weight matrix
        self.Wbo = np.random.normal(0,1,[n_out,1]) #bias-outout weight matrix
        self.Wro = np.random.normal(0,1,[n_out,neu]) #reservoir-output weight matrix, the one parameter to be trained
        self.leakrate = gama #the network's leak rate
        self.ro = ro #the network's desired spectral radius
        self.psi = psi #the network's sparcity, in 0 to 1 notation
        self.in_scale = in_scale #the scaling of Wir.
        self.bias_scale = bias_scale #the scaling of Wbr
        self.alfa = alfa #learning rate of the Recursive Least Squares Algorithm
        self.forget = forget #forget factor of the RLS Algorithm

        self.Wrr = Sparcity(self.Wrr,self.psi) #the probability of a memeber of the Matrix Wrr being zero is psi.

        #forcing Wrr to have ro as the maximum eigenvalue
        eigs = np.linalg.eigvals(self.Wrr)
        radius = np.abs(np.max(eigs))
        self.Wrr = self.Wrr/radius
        self.Wrr = self.Wrr*ro

        #scaling
        self.Wbr = bias_scale*self.Wbr
        self.Wir = in_scale*self.Wir



        #initial conditions.

        self.a = np.zeros([neu,1])

        #covariance matrix
        self.P = np.eye(neu)/alfa

    



    def Train(self,ref):
        #ref e o vetor de todas as saidas desejados no dado instante de tempo.
        for saida in range(self.n_out):
            #Transpose respective output view..
            Theta = Wro[saida,:]
            Theta = Theta.reshape([self.neu,1])

            #MQR equations

            #the P equation step by step, it gets ugly if you do it at once
            A = self.P/self.forget
            B = np.dot(self.P,self.a)
            C = np.dot(B,self.a.reshape([1,self.neu]))
            D = np.dot(C,self.P)
            E = dot(self.a.reshape([1,self.neu]),self.P)
            F = self.forget + dot(E,self.a)

            #atualizacao final
            self.P = A - D/(self.forget*F)

            #calculo do erro
            e = Wro[saida,:]*a - ref[saida]

            Theta = Theta - e*dot(self.P,self.a)

            Theta = Theta.reshape([1,self.neu])

            Wro[saida,:] = Theta



        #rotina de treinamento atraves dos minimos quadrados recursivos atualiza Wor.

    def Update(self):
        pass
        #rotina de atualizacao dos estados, retorna a saida.


RNN(100,1,1)




