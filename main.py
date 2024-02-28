import numpy as np
u,v,w = np.loadtxt("velocityfld_ascii.dat",skiprows=2,usecols=(3,4,5),unpack=True)
velocity_fld = []
velocity_fld.append(u)
velocity_fld.append(v)
velocity_fld.append(w)

from Energy_Spectrum import compute_Ek_spectrum
k,E = compute_Ek_spectrum(velocity_fld,reference_velocity=(0.1*np.sqrt(1.4)))

import matplotlib.pyplot as plt

fig = plt.figure()
plt.title("Kinetic Energy Spectrum")
plt.xlabel(r"k (wavenumber)")
plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
plt.ylim([1.0e-9,1.0e-1])
plt.loglog(k,E)
plt.show()