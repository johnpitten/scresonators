import numpy as np

def DCM_S21(freqs, f0, Q, Qc, phi):
    s21 = 1-(Q*np.exp(-1j*phi)/Qc)/(1+2j*Q*(freqs-f0)/f0)
    return s21

#generate sample data
f0 = 4.5 #GHz
Qi = 100000
Qc = 150000
phi = 0.1
Qtot = 1/(1/Qi+np.cos(phi)/Qc)
linewidth = 2*f0/(Qtot)
delay =  0#ns


fstart = f0-10*linewidth
fstop = f0+10*linewidth
npoints = 501
freqs = np.linspace(fstart, fstop, npoints)


S21 = np.exp(-2j*np.pi*freqs*delay)*DCM_S21(freqs, f0, Qtot, Qc, phi)
