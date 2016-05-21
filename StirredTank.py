__author__ = 'jean'
import numpy as np

import matplotlib.pyplot as plt

from copy import *

from RNN import *

def FindEquilibrium_q(T):

    Ti = 15.0 #temperatura de entrada
    Q = 1100.0 #Calor
    ro = 1.0 #densidade
    cp = 4186.0 #coeficiente calorifico

    invq = (T - Ti)*ro*cp/Q

    q = 1/invq

    return q

def FindEquilibrium_T(q):

    Ti = 15.0 #temperatura de entrada
    Q = 1100.0 #Calor
    ro = 1.0 #densidade
    cp = 4186.0 #coeficiente calorifico


    return Ti + Q/(q*ro*cp)

def delay_of_signal(signal,steps_ahead = 23): #less powerful version of dataset_time_window, for testing purposes.
    signal = np.array(signal)
    signal.shape = (len(signal), 1)
    signal_output = signal[steps_ahead:]
    signal_input = signal[:-steps_ahead]
    return signal_input,signal_output

def Euler(step,f,x,u):
    xnext = x + step*f(x,u)

    return xnext



def StirredTank(T,q):
    Vtank = 1.13 #volume do tanque
    Ti = 15.0 #temperatura de entrada
    Q = 1100.0 #Calor
    ro = 1.0 #densidade
    cp = 4186.0 #coeficiente calorifico
    f = (q*(Ti-T) + Q/(ro*cp))/Vtank #Equacao diferencial em tempo continuo do balanco de energia do tanque
    return f


def Tube(Ttube,Ttank):

    Tau = 29.0 #constante de tempo do tubo
    K = 0.99 #ganho do tubo
    f = K*(Ttank - Ttube)/Tau #equacao de estados do tubo
    return f

def RungeKutta4(step,f,x,u): #runge kutta

    k1 = f(x,u)
    k2 = f(x + step*k1/2.0,u)
    k3 = f(x + step*k2/2.0,u)
    k4 = f(x + step*k3,u)

    xnext = x + step * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    return xnext

def UpdateTimeWindow(A,x): #Dado um array unidimensional A, elimina A[1] e acrescenta A[len(A)-1] = x
    B = A[1:]
    B = np.append(B,x)
    return B

def normalize(q):
    qmin = 0.005
    qmax = 0.03
    x = (q-qmin)/(qmax - qmin)

    return x

def denormalize(x):

    qmin = 0.005
    qmax = 0.03
    q = (qmax - qmin)*x + qmin


    return q

#Parametros do problema.
t_Stop = 700 #tempo corrente da simulacao
t_step = 0.5 #periodo de amostragem da simulacao.
t_Ctrl = 4 #Periodo de amostragem do controle.
Vtube = 1.02 #volume no tubo de transporte

T0 = 0 #condicao inicial

t = np.arange(0,t_Stop,t_step) #tempo de simulacao

iterations = range(len(t))
Q_plot = np.empty_like(t)

T_plot = np.empty_like(t)

Ttube_plot = np.empty_like(t)
Tout_plot = np.empty_like(t)

T = T0
Ttube = 0.99*T0
Tout = Ttube
#Q = 0.0167 +0.01*np.sin(t)
q = 0.0167

qmax = 0.03

qmin = 0.005

Tmax = FindEquilibrium_T(qmin)

Tmin = FindEquilibrium_T(qmax)

# A Rede
delta = 30
delta_converted = int(round(delta*t_Ctrl/t_step))
neurons = 500
inputs = 2
outputs = 1
gama = 0.5
ro = 1
psi = 0.5
f_ir = 0.1
f_br = 0.5

CtrlNetTrainer = rNN(neurons,inputs,outputs,gama,ro,psi,f_ir,f_br)
CtrlNetController = deepcopy(CtrlNetTrainer)


#Criacao da janela da condicao inicial.

q_past = np.empty_like(np.arange(delta_converted))
T_past = np.empty_like(q_past)
Ttube_past =  np.empty_like(q_past)
Tout_past = np.empty_like(q_past)
Tfuturo = 40 #referencia degrau.
Tref_plot = np.empty_like(t)
Training_Error_plot = np.empty_like(t)

Control_plot = np.empty_like(t)

for i in range(delta_converted):#Numero de passos necessarios para dar o equivalente a passagem no tempo de delta.

    T_past[i] = T
    Ttube_past[i] = Ttube
    q_past[i] = q
    T = RungeKutta4(t_step,StirredTank,T,q)
    Ttube = RungeKutta4(t_step,Tube,Ttube,T)
    if i > np.floor(Vtube/(q*t_step)):

        Tout = Ttube_past[i - np.floor(Vtube/(q*t_step))]


    Tout_past[i] = Tout + np.random.randn()
    q = 0.0167

for i in iterations:
    #espaco para definicao do sinal de controle, onde entraria a rede
    #q = Q[i]
    if i%(t_Ctrl/t_step) == 0:
    #etapa de treino

        #Tmean = np.mean(Tout_past)
       # Tstd = np.std(Tout_past)


        white_noise = np.random.randn()
        CtrlNetTrainer.Update([(Tout_past[0]-Tmin)/(Tmax - Tmin),((Tout + white_noise)-Tmin)/(Tmax - Tmin)])
        #CtrlNetTrainer.Update([(Tout_past[0]-Tmean)/Tstd,((Tout + white_noise)-Tmean)/Tstd])
        u_ctrlpast = normalize(q_past[0])
        CtrlNetTrainer.Train(u_ctrlpast)
        trainingerror = CtrlNetTrainer.trainingError(u_ctrlpast)


    #etapa de teste
        CtrlNetController.CopyWeights(CtrlNetTrainer)
        uctrl = CtrlNetController.Update([((Tout + white_noise)-Tmin)/(Tmax-Tmin),(Tfuturo-Tmin)/(Tmax-Tmin)])
        q = denormalize(uctrl)
        #q = CtrlNetController.Update([((Tout + white_noise)-Tmean)/Tstd,(Tfuturo-Tmean)/Tstd])

        control = q

        if q > qmax:
            q = qmax

        if q < qmin:
           q = qmin



    Training_Error_plot[i] = trainingerror
    #fim do espaco para a lei de controle
    T_plot[i] = T
    Ttube_plot[i] = Ttube
    Q_plot[i] = q
    T = RungeKutta4(t_step,StirredTank,T,q)
    Ttube = RungeKutta4(t_step,Tube,Ttube,T)
    if i > np.floor(Vtube/(q*t_step)):

        Tout = Ttube_plot[i - round(Vtube/(q*t_step))]


    Tout_plot[i] = Tout
    Tref_plot[i] = Tfuturo
    Control_plot[i] = control

    q_past = UpdateTimeWindow(q_past,q)
    Tout_past = UpdateTimeWindow(Tout_past,Tout)
    T_past = UpdateTimeWindow(T_past,T)






f, sub = plt.subplots(4, sharex=True)



p1 = sub[0].plot(t,T_plot,label = 'Temperatura do tanque')



p2 = sub[1].plot(t,Q_plot,label = 'Vazao Inlet')

p3 = sub[0].plot(t,Ttube_plot,label = 'Temperatura Tubo')

p4 = sub[0].plot(t,Tout_plot,label = 'Temperatura Saida')

p5 = sub[0].plot(t,Tref_plot,label = 'Referencia')

p6 = sub[2].plot(t,Training_Error_plot,label = 'erro de treinamento')

p7 = sub[3].plot(t,Control_plot,label = 'acao de controle')

sub[0].legend()

sub[1].legend()

sub[2].legend()

sub[3].legend()

plt.show(p1)

plt.show(p2)

plt.show(p3)

plt.show(p4)







