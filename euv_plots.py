import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

euva = np.loadtxt('../MavenProcessing/Output/euv_A.csv', delimiter=',', unpack=True)
euvb = np.loadtxt('../MavenProcessing/Output/euv_B.csv', delimiter=',', unpack=True)
euvc = np.loadtxt('../MavenProcessing/Output/euv_C.csv', delimiter=',', unpack=True)

euva[0] = np.floor(euva[0])
euvb[0] = np.floor(euvb[0])
euvc[0] = np.floor(euvc[0])


idx_a = np.where(euva[0]==2349)[0]
idx_b = np.where(euvb[0]==2349)[0]
idx_c = np.where(euvc[0]==2349)[0]

f, axes = plt.subplots(3,2)

axes[0, 0].plot(euva[0], euva[1])
axes[1, 0].plot(euvb[0], euvb[1])
axes[2, 0].plot(euvc[0], euvc[1])
axes[0, 0].vlines(2349, euva[1].min(), euva[1].max())
axes[1, 0].vlines(2349, euvb[1].min(), euvb[1].max())
axes[2, 0].vlines(2349, euvc[1].min(), euvc[1].max())

axes[0, 1].hist(euva[1], bins=150)
axes[1, 1].hist(euvb[1], bins=150)
axes[2, 1].hist(euvc[1], bins=150)
axes[0, 1].vlines(euva[1, idx_a], 0, 100)
axes[1, 1].vlines(euvb[1, idx_b], 0, 100)
axes[2, 1].vlines(euvc[1, idx_c], 0, 100)


axes[1, 1].vlines(0.00022, 0, 100, alpha=0.5)
axes[1, 1].vlines(0.00013, 0, 100, alpha=0.5)
plt.savefig('Output/euv.pdf')
