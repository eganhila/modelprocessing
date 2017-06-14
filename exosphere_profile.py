import numpy as np
import matplotlib.pyplot as plt

mars_r = 3390.0
alt = np.linspace(0, 1*mars_r, 250)


O_hot = 5.23e3*np.exp(-alt/626.2)+9.76e2*np.exp(-alt/2790) +3.71e4*np.exp(-alt/88.47)
O_cold = 5.85e13*np.exp(-alt/10.56)+7.02e9*np.exp(-alt/33.97)
O_total = O_hot + O_cold

H_profile = 1.5e5*np.exp(25965*(1/(alt+3393.5)-1/3593.5)) 


#plt.subplot(121)
plt.plot(O_hot, alt, ls='--', color="MidnightBlue", label='O hot')
plt.plot(O_cold, alt, ls=':', color="MidnightBlue", label='O cold')
plt.plot(O_total, alt, color="MidnightBlue", label='O total')
plt.plot(H_profile, alt, color="Crimson", label='H total')
plt.xlim(1E2,1e10)
plt.ylabel("Altitude")
#plt.semilogy()
plt.loglog()
plt.xlabel("Number Density ($cm^{-3}$)")
plt.legend(loc='upper right')


#plt.subplot(122)
#plt.loglog()
#plt.semilogy()
plt.gcf().set_size_inches(4,6)
plt.tight_layout()

plt.savefig('Output/exosphere.pdf')
