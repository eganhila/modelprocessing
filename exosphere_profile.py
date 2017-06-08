import numpy as np
import matplotlib.pyplot as plt

mars_r = 3390.0
alt = np.linspace(0, 2*mars_r, 250)


O_hot = 5.23e3*np.exp(-alt/626.2)+9.76e2*np.exp(-alt/2790) +3.71e4*np.exp(-alt/88.47)
O_cold = 5.85e13*np.exp(-alt/10.56)+7.02e9*np.exp(-alt/33.97)
O_total = O_hot + O_cold

H_profile = 1.5e5*np.exp(25965*(1/(alt+3393.5)-1/3593.5)) 


plt.subplot(121)
plt.plot(alt, O_hot)
plt.plot(alt, O_cold)
plt.plot(alt, O_total)
plt.semilogy()


plt.subplot(122)
plt.plot(alt, H_profile)
plt.semilogy()

plt.savefig('Output/exosphere.pdf')
