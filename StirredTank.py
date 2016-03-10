__author__ = 'jean'
import numpy as np

import matplotlib.pyplot as plt


def StirredTank(T,q):
    Vtank = 1.13
    Ti = 15
    Q = 1100
    ro = 1
    cp = 4186
    f = (q*(Ti-T) + Q/(ro*cp))/Vtank
    return f


def Tube(Ttube,Ttank):

    Tau = 29
    K = 0.99
    f = K*(Ttank - Ttube)/Tau
    return f

def RungeKutta4(step,f,x,u):

    k1 = f(x,u)
    k2 = f(x+step*k1/2,u)
    k3 = f(x+step*k2/2,u)
    k4 = f(x+step*k3,u)

    xnext = x + step/6*(k1+2*k2+2*k3+k4)

    return xnext





t_step = 0.1
Vtube = 1.02

T0 = 0

t = np.arange(0,600,t_step)

iterations = range(len(t))
Q_plot = np.empty_like(t)

T_plot = np.empty_like(t)

Ttube_plot = np.empty_like(t)
Tout_plot = np.empty_like(t)

T = T0
Ttube = 0.99*T0
Tout = Ttube
Q = 0.025 +0.02*np.sin(t)
q = 0.0167


for i in iterations:

    #espaco para definicao do sinal de controle, onde entraria a rede
    q = Q[i]


    #fim do espaco para a lei de controle

    T = RungeKutta4(t_step,StirredTank,T,q)
    Ttube = RungeKutta4(t_step,Tube,Ttube,T)
    T_plot[i] = T
    Ttube_plot[i] = Ttube
    Q_plot[i] = q
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






