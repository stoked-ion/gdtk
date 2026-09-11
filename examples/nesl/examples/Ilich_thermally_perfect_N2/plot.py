import matplotlib.pyplot as plt
import numpy as np

nesl = np.loadtxt('ilich.dat', delimiter=',', skiprows=1)
x_nesl = (nesl[:, 0].max() - nesl[:, 0]) * 1000 + 12.7
T_nesl = nesl[:, 3]

cfdpp = np.loadtxt('cfdpp.csv', delimiter=',')
x_cfdpp = cfdpp[:,0][np.argsort(cfdpp[:,0])]
T_cfdpp = cfdpp[:,1][np.argsort(cfdpp[:,0])]

drnse = np.loadtxt('drnse.csv', delimiter=',')
x_drnse = drnse[:,0][np.argsort(drnse[:,0])]
T_drnse = drnse[:,1][np.argsort(drnse[:,0])]

plt.figure(figsize=(6,6))

plt.plot(x_drnse, T_drnse, label="DRNSE")
plt.plot(x_cfdpp, T_cfdpp, label="CFD++")
plt.plot(x_nesl, T_nesl, label="NESL")

plt.ylim(0, 1400)
plt.xlim(14.5, 12.7)
plt.legend(loc="lower right")
plt.xlabel("Radius, mm")
plt.ylabel("Temperature, K")
plt.tight_layout()

plt.savefig("ilich.png", dpi=300)
plt.savefig("ilich.pdf", format='pdf', dpi=300, bbox_inches="tight")
