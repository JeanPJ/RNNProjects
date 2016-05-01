__author__ = 'jean'
import numpy as np
import matplotlib.pyplot as plt

from RNN import *

def dataset_time_window(signal, window_size = 2, steps_ahead = 23):
    start_index = window_size-1
    signal = np.array(signal)
    signal.shape = (len(signal), 1)
    signal_output = signal[start_index + steps_ahead:]
    signal_input = signal[start_index:-steps_ahead]
    for k in np.arange(1, window_size):
        signal_input = np.concatenate((signal_input, signal[start_index - k:-steps_ahead - k]), axis=1)
    return signal_input, signal_output


def delay_of_signal(signal,steps_ahead = 23): #less powerful version of dataset_time_window, for testing purposes.
    signal = np.array(signal)
    signal.shape = (len(signal), 1)
    signal_output = signal[steps_ahead:]
    signal_input = signal[:-steps_ahead]
    return signal_input,signal_output


n = np.arange(0,1000, 0.1)

u = np.sin(n)
Rede = rNN(100,1,1)

u, d = delay_of_signal(u)
y = np.empty_like(n)

for i in range(len(u)):
    Rede.Train(d[i])
    y[i] = Rede.Update(u[i])



plt.plot(d[:1000],'.-g', linewidth = 3), plt.plot(y[:1000],'.-'), plt.show()





