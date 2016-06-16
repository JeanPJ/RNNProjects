__author__ = 'jean'
import numpy as np

import matplotlib.pyplot as plt

from copy import *

from RNN import *

from scipy import signal


def FindEquilibrium_q(T):
    Ti = 15.0  # temperatura de entrada
    Q = 1100.0  # Calor
    ro = 1.0  # densidade
    cp = 4186.0  # coeficiente calorifico

    invq = (T - Ti) * ro * cp / Q

    q = 1 / invq

    return q


def FindEquilibrium_T(q):
    Ti = 15.0  # temperatura de entrada
    Q = 1100.0  # Calor
    ro = 1.0  # densidade
    cp = 4186.0  # coeficiente calorifico

    return Ti + Q / (q * ro * cp)


def delay_of_signal(signal, steps_ahead=23):  # less powerful version of dataset_time_window, for testing purposes.
    signal = np.array(signal)
    signal.shape = (len(signal), 1)
    signal_output = signal[steps_ahead:]
    signal_input = signal[:-steps_ahead]
    return signal_input, signal_output


def Euler(step, f, x, u):
    xnext = x + step * f(x, u)

    return xnext


def StirredTank(T, q):
    Vtank = 1.13  # volume do tanque
    Ti = 15.0  # temperatura de entrada
    Q = 1100.0  # Calor
    ro = 1.0  # densidade
    cp = 4186.0  # coeficiente calorifico
    f = (q * (Ti - T) + Q / (ro * cp)) / Vtank  # Equacao diferencial em tempo continuo do balanco de energia do tanque
    return f


def Tube(Ttube, Ttank):
    Tau = 29.0  # constante de tempo do tubo
    K = 0.99  # ganho do tubo
    f = K * (Ttank - Ttube) / Tau  # equacao de estados do tubo
    return f


def RungeKutta4(step, f, x, u):  # runge kutta

    k1 = f(x, u)
    k2 = f(x + step * k1 / 2.0, u)
    k3 = f(x + step * k2 / 2.0, u)
    k4 = f(x + step * k3, u)

    xnext = x + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return xnext


def UpdateTimeWindow(A, x):  # Dado um array unidimensional A, elimina A[1] e acrescenta A[len(A)-1] = x
    B = A[1:]
    B = np.append(B, x)
    return B


def normalize(q, mn, std):
    x = (q - mn) / std

    return x


def denormalize(x, mn, std):
    q = std * x + mn

    return q


def lowpass(x, u):
    w0 = 5
    f = u - w0 * x
    return f


def RedNoise(x, t_step):
    white_noise = np.random.randn()
    xnext = RungeKutta4(t_step, lowpass, x, white_noise)
    return xnext


def RandomStair(t, stair_step,
                values):  # vetor que gera uma sequencia aleatoria de valores, onde t eh um vetor de referencia, stair_step eh o numero de iteracoes que se mantem um valor
    # e values sao os valores que o degrau pode assumir
    values = np.array(values)
    Stair = np.empty_like(t)
    idx = 0
    for i in range(Stair.size):
        if i % stair_step == 0:
            idx = np.random.randint(0, len(values))  # escolhe o indice aleatoriamente a cada passo da escada.
        Stair[i] = values[idx]

    return Stair


# Parametros do problema.
step_stop = 6000  # numero de passos totais, caso queira se usar essa metrica, no ponto de vista do controlador.
t_step = 4  # periodo de amostragem da simulacao.

First_step_end = 2000
Vtube = 1.02  # volume no tubo de transporte
q0 = 0.0167
T0 = FindEquilibrium_T(q0)  # condicao inicial

t = np.arange(0, step_stop * t_step, t_step, dtype=np.float64)  # tempo de simulacao

iterations = range(len(t))
Q_plot = np.empty_like(t)

T_plot = np.empty_like(t)

Ttube_plot = np.empty_like(t)
Tout_plot = np.empty_like(t)

T = T0
Ttube = 0.99 * T0
Tout = Ttube
# Q = 0.0167 +0.01*np.sin(t)
q = q0

qmax = 0.03

qmin = 0.005

qmn = 0.0250
qstd = 0.0175

Tmax = FindEquilibrium_T(qmin)

Tmin = FindEquilibrium_T(qmax)

# A Rede
delta = 30
neurons = 500
inputs = 2
outputs = 1
gama = 0.49
ro = 1
psi = 0.5
f_ir = 0.08
f_br = 0.5

CtrlNetTrainer = rNN(neurons, inputs, outputs, gama, ro, psi, f_ir, f_br)
CtrlNetController = deepcopy(CtrlNetTrainer)

Wro_plot = np.empty((len(t), neurons), dtype=float)


# Criacao da janela da condicao inicial.

q_past = np.empty_like(np.arange(delta))
T_past = np.empty_like(q_past)
Ttube_past = np.empty_like(q_past)
Tout_past = np.empty_like(q_past)
Tfuturo = 0
Tref_plot = np.empty_like(t)
Training_Error_plot = np.empty_like(t)

Control_plot = np.empty_like(t)

# define vetor de referencias
# for i in range(Tref_plot.size):
#    if i < int(round(First_step_end/t_step)):
#        Tfuturo = RedNoise(Tfuturo,t_step)
#        Tref_plot[i] = 100*Tfuturo + 30
#    else:
#        break
stair_step_ctrl = 200
Tref_plot[:First_step_end] = 35 + 30 * signal.square(2 * np.pi * 0.001 * t[:First_step_end])
Tref_plot[First_step_end:] = RandomStair(Tref_plot[First_step_end:], stair_step_ctrl,
                                         [42, 40, 50, 60, 55, 45, 40, 36, 34, 37, 39, 49])

# Tref_plot = RandomStair(Tref_plot,400,[30,40,50,60,55,45,40,35,30,32,33,49])

Tmn = np.mean(Tref_plot)

Tstd = np.std(Tref_plot)








# uma rodada em malha aberta
for i in range(delta):  # Numero de passos necessarios para dar o equivalente a passagem no tempo de delta.

    T_past[i] = T
    Ttube_past[i] = Ttube
    q_past[i] = q
    T = RungeKutta4(t_step, StirredTank, T, q)
    Ttube = RungeKutta4(t_step, Tube, Ttube, T)
    if i > np.floor(Vtube / (q * t_step)):
        Tout = Ttube_past[i - int(round(Vtube / (q * t_step)))]

    Tout_past[i] = Tout + np.random.randn()
    q = q0

for i in range(step_stop):
    # espaco para definicao do sinal de controle, onde entraria a rede
    # q = Q[i]
    # etapa de treino

    # Tmean = np.mean(Tout_past)
    # Tstd = np.std(Tout_past)


    white_noise = np.random.randn()
    CtrlNetTrainer.Update([normalize(Tout_past[0], Tmn, Tstd), normalize(Tout, Tmn, Tstd)])
    # CtrlNetTrainer.Update([(Tout_past[0]-Tmean)/Tstd,((Tout + white_noise)-Tmean)/Tstd])
    u_ctrlpast = normalize(q_past[0], qmn, qstd)
    CtrlNetTrainer.Train(u_ctrlpast)
    trainingerror = CtrlNetTrainer.trainingError(u_ctrlpast)


    # etapa de teste
    CtrlNetController.CopyWeights(CtrlNetTrainer)
    uctrl = CtrlNetController.Update([normalize(Tout, Tmn, Tstd), normalize(Tref_plot[i], Tmn, Tstd)])
    # uctrl = CtrlNetController.Update([((Tout + white_noise)-Tmean)/Tstd,(Tfuturo-Tmean)/Tstd])
    q = denormalize(uctrl, qmn, qstd)
    control = q

    if q > qmax:
        q = qmax

    if q < qmin:
        q = qmin

    Wro_plot[i] = CtrlNetController.getWro(0)

    Training_Error_plot[i] = trainingerror
    # fim do espaco para a lei de controle
    T_plot[i] = T
    Ttube_plot[i] = Ttube
    Q_plot[i] = q
    T = RungeKutta4(t_step, StirredTank, T, q)
    Ttube = RungeKutta4(t_step, Tube, Ttube, T)
    if i > np.floor(Vtube / (q * t_step)):
        Tout = Ttube_plot[i - int(round(Vtube / (q * t_step)))]

    Tout_plot[i] = Tout
    # Tref_plot[i] = Tfuturo
    Control_plot[i] = control

    q_past = UpdateTimeWindow(q_past, q)
    Tout_past = UpdateTimeWindow(Tout_past, Tout)
    T_past = UpdateTimeWindow(T_past, T)


Wro_trueplot = np.mean(np.diff(Wro_plot**2),1)

f, sub = plt.subplots(4, sharex=True)



# p1 = sub[0].plot(t,T_plot,label = 'Temperatura do tanque')



p2 = sub[1].plot(iterations, Q_plot, label='Vazao Inlet')

# p3 = sub[0].plot(t,Ttube_plot,label = 'Temperatura Tubo')

p4 = sub[0].plot(iterations, Tout_plot, label='Temperatura Saida')

p5 = sub[0].plot(iterations, Tref_plot, label='Referencia')

p6 = sub[2].plot(iterations, Wro_trueplot, label='mean(diff(Uw)^2)')

p7 = sub[3].plot(iterations, Control_plot, label='acao de controle')

sub[0].legend()

sub[1].legend()

sub[2].legend()

sub[3].legend()

# plt.show(p1)

plt.show(p2)

# plt.show(p3)

plt.show(p4)
