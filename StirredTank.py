__author__ = 'jean'
import numpy as np

import matplotlib.pyplot as plt



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





t_step = 0.1 #periodo de amostragem, nao funciona coms
Vtube = 1.02 #volume no tubo de transporte

T0 = 0 #condicao inicial

t = np.arange(0,600,t_step) #tempo de simulacao

iterations = range(len(t))
Q_plot = np.empty_like(t)

T_plot = np.empty_like(t)

Ttube_plot = np.empty_like(t)
Tout_plot = np.empty_like(t)

T = T0
Ttube = 0.99*T0
Tout = Ttube
Q = 0.0167 +0.01*np.sin(t)
q = 0.0167


for i in iterations:

    #espaco para definicao do sinal de controle, onde entraria a rede
    q = Q[i]


    #fim do espaco para a lei de controle
    T_plot[i] = T
    Ttube_plot[i] = Ttube
    Q_plot[i] = q
    T = RungeKutta4(t_step,StirredTank,T,q)
    Ttube = RungeKutta4(t_step,Tube,Ttube,T)
    if i > np.floor(Vtube/(q*t_step)):

        Tout = Ttube_plot[i-np.floor(Vtube/(q*t_step))]


    Tout_plot[i] = Tout






p1 = plt.plot(t,T_plot,label = 'Temperatura do tanque')



p2 = plt.plot(t,Q_plot,label = 'Vazao Inlet')

p3 = plt.plot(t,Ttube_plot,label = 'Temperatura Tubo')

p4 = plt.plot(t,Tout_plot,label = 'Temperatura Saida')

plt.legend()

plt.show(p1)

plt.show(p2)

plt.show(p3)

plt.show(p4)






