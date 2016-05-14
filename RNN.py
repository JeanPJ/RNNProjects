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
class rNN:

    def __init__(self,neu,n_in,n_out,gama=0.5,ro=1,psi=0.5,in_scale=0.1,bias_scale=0.5,alfa=10.0,forget = 1.0-10.0**-6.0):
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

    

    def trainingError(self,ref):

        Ref = np.array(ref)
        if self.n_out > 1:
            Ref = Ref.reshape(len(ref),1)

        return np.dot(self.Wro,self.a)+ self.Wbo - Ref

    def Train(self,ref):
        #ref e o vetor de todas as saidas desejados no dado instante de tempo.
        #calcular o vetor de erros
        e = self.trainingError(ref)

        for saida in range(self.n_out):
            #Transpose respective output view..
            Theta = self.Wro[saida,:]
            Theta = Theta.reshape([self.neu,1])

            #MQR equations

            #the P equation step by step, it gets ugly if you do it at once
            A = self.P/self.forget
            B = np.dot(self.P,self.a)
            C = np.dot(B,self.a.reshape([1,self.neu]))
            D = np.dot(C,self.P)
            E = np.dot(self.a.reshape([1,self.neu]),self.P)
            F = self.forget + np.dot(E,self.a)

            #atualizacao final
            self.P = A - D/(self.forget*F)

            #calculo do erro
            Theta = Theta - e[saida]*np.dot(self.P,self.a)

            Theta = Theta.reshape([1,self.neu])

            self.Wro[saida,:] = Theta



        #rotina de treinamento atraves dos minimos quadrados recursivos atualiza Wor.

    def Update(self,input):
        Input = np.array(input)
        Input = Input.reshape(Input.size,1)
        if Input.size == self.n_in:
            self.a = (1-self.leakrate)*self.a + self.leakrate*np.tanh(np.dot(self.Wrr,self.a) + np.dot(self.Wir,Input) + self.Wbr)
            y = np.dot(self.Wro,self.a) + self.Wbo
            return y
        else:
            raise ValueError("input must have size n_in")



    def CopyWeights(self, Rede):
        if self.Wro.shape == Rede.Wro.shape:
            self.Wro = np.copy(Rede.Wro)
        else:
            print "shapes of the weights are not equal"



        #rotina de atualizacao dos estados, retorna a saida.


#oi = rNN(100,1,1)

#oi.Train(1)

#t = np.arange(0,600,0.01)

#Y = np.sin(t)

#plt.plot(t,Y)

#plt.show()


#oi.Update(2)




