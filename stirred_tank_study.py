__author__ = 'jean'


import numpy as np
import scipy as sp
import matplotlib.pyplot as mp


#AhdTh/dt = q(Ti - T) + Q/(ro*cp)

#Como a vazoo e sempre positiva, o tanque sempre esta estavel.


Ti = 15
Q = 1100
Vtank = 1.13
ro = 1
cp = 4186

q = np.arange(0.005,0.03,0.0001)

T = Ti + Q/(q*ro*cp)


mp.figure
mp.plot(q,T)

mp.show()
