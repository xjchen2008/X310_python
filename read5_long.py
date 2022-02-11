import functions as fn
from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fftshift
plt.rc('font', size=15)

N=1000#400#*400#4000#1000*40#40000#1120000*2
M = 100
c = 3e8
fs = 200e6
bw = 100e6
Tp = N/fs
#Fp0 = 500e3/10*2
rubish =2000 # + 120#50*N #10000 #1000 #8000*50#*1000# *9#256*8
read_bin = fn.ReadBin()
readbin2 = read_bin.readbin2
sig = readbin2("/home/james/project/uhd/host/build/examples/data_backup/usrp_samples_loopback_0.dat",N, rubish)
tx = readbin2("/home/james/project/uhd/host/build/examples/usrp_samples.dat",N,0)
#tx = readbin2("/home/james/project/uhd/host/build/examples/mypython/usrp_samples_init.dat",N,0)

rx = sig
# pc calculation 1
pc1 = fn.PulseCompr(rx=rx, tx=tx, win=1, unit='linear')
pc1_db = (20 * np.log10(abs(pc1)))
pc1_db_normalized = pc1_db - pc1_db.max()
# pc calculation 2
#pc2 = fn.PulseCompr(rx=rx_woo, tx=tx_woo, win=1, unit='linear')
#pc2_db = (20 * np.log10(abs(pc2)))
#pc2_db_normalized = pc2_db - pc2_db.max()


freq = np.fft.fftfreq(len(rx), d=1. / fs)
offset_cal =0 #98 # a calibration distance due to Ettus delay and cable delay
#distance = c * freq /(2*M*np.power(Fp0, 2)) *4 - offset_cal

Rmax1 = len(rx) * c / (2 * fs)  # 1/del_F *c/2
distance = np.linspace(0, Rmax1, len(rx)) -offset_cal # FMCW PC=Matched Filter radar

#distance = c / 2 * freq / (bw / Tp) - offset_cal # FMCW PC = Stretch method
plt.figure()


#plt.plot((distance), (pc1_db), 'b*-',(distance), (pc2_db), 'r*-')  # Matched Filter PC
#plt.plot(fftshift(pc1_db_normalized), 'k*-')
plt.plot(fftshift(distance), fftshift(pc1_db), 'b*-')
#plt.xlabel('Sample Number')
plt.xlabel('Distance [m]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.xlim([-20, 2000])
#plt.ylim([-50, 10])
plt.legend(['chirp','p4'])
plt.title('Pulse Compression')


x = np.linspace(0, len(sig)-1, len(sig))
plt.figure()
plt.plot(x, 32767*np.imag(sig), 'bo-', x, 32767*np.real(sig), 'yo-')
plt.title('Time Domain Received Signal')
plt.legend(['Imag', 'Real'])
plt.grid()
plt.figure()
plt.plot(x, 32767*np.imag(tx), 'bo-',x, 32767*np.real(tx),'yo-')
plt.title('Time Domain Transmitted Signal')
plt.legend(['Imag', 'Real'])
plt.grid()

plt.show()