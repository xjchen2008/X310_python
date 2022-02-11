import functions as fn
import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from numpy.fft import fftshift, fft, ifft
plt.rc('font', size=15)


def get_gamma_m(v_rx, v_tx):
    # Read a data file and return measured reflection coefficient
    gamma_m = fft(v_rx) / fft(v_tx)
    return gamma_m


def one_port_cal(v_tx, N, v_rx_short, v_rx_open, v_rx_load):
    # https://www.rfmentor.com/sites/default/files/NA_Error_Models_and_Cal_Methods.pdf
    # Short, open, load one-port calibration and 3-term error model
    # Assume that gamma1 = -1; gamma2 = 1, gamma3 = 0.
    read_bin = fn.ReadBin()
    readbin2 = read_bin.readbin2
    gamma_m1 = get_gamma_m(v_rx_short, v_tx)
    gamma_m2 = get_gamma_m(v_rx_open, v_tx)
    gamma_m3 = get_gamma_m(v_rx_load, v_tx)
    e00 = gamma_m3
    e11 = (gamma_m2 + gamma_m1 - 2 * gamma_m3) / (gamma_m2 - gamma_m1)
    delta_e = e00 + gamma_m2 * e11 - gamma_m2
    return e00, e11, delta_e


def main(v_rx, v_tx, v_rx_short, v_rx_open, v_rx_load, N=1000):
    n = np.linspace(0,N-1,N)
    v_tx = v_tx
    v_rx = v_rx

    # step 1: get error s-parameter and measured reflection coefficient
    e00, e11, delta_e = one_port_cal(v_tx, N,v_rx_short=v_rx_short,v_rx_open=v_rx_open,v_rx_load=v_rx_load)

    gamma_m = get_gamma_m(v_rx, v_tx) # read rx data file and calculate reflection coe in freq domain

    # step 2: get the actual/corrected/calibrated reflection coefficient by correcting the measured one and
    # error-s-parameter
    gamma = (gamma_m - e00) / (gamma_m * e11 - delta_e)

    # step 3: get the calibrated time-domain received signal
    v_rx_cal = ifft(gamma*fft(v_tx))
    '''
    plt.plot(n, v_rx_cal/v_tx)
    plt.ylim([-2,2])
    plt.title('Time Domain Calibrated $\Gamma$')
    plt.xlabel('Sample number')
    plt.ylabel('$\Gamma magnitude in linear unit')
    plt.grid()
    plt.figure()
    plt.plot(n, v_rx_cal.real, n, v_rx_cal.imag)
    plt.title('Time Domain Calibrated Signal')
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.figure()
    plt.plot(n, v_rx.real, n, v_rx.imag)
    plt.title('Time Domain Original Signal')
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.figure()
    coe = fn.Coe(fc=50e6, bw=20e6, fs=200e6, N=1000)
    freq= coe.freq
    fn.plot_freq_db(freq, v_rx_cal,color='b', normalize=True )
    fn.plot_freq_db(freq, v_rx, color='r', normalize=True)
    #plt.figure()
    #fn.plot_freq_phase(freq, v_rx_cal,color='b' )
    #fn.plot_freq_phase(freq, v_rx, color='r')

    pc0 = fn.PulseCompr(rx=v_tx, tx=v_tx, win=1, unit='log')
    pc1 = fn.PulseCompr(rx=v_rx,tx=v_tx, win=1,unit='log')
    pc2 = fn.PulseCompr(rx=v_rx_cal, tx=v_tx, win=1, unit='log')
    freq = fn.Coe().freq
    plt.figure()
    plt.plot(pc0)
    plt.plot(pc1)
    plt.plot(pc2)
    
    plt.show()
    #'''
    return v_rx_cal


if __name__ == '__main__':
    N = 1000
    Navg = 1
    n=np.linspace(0,N-1,N)
    read_bin = fn.ReadBin()
    readbin2 = read_bin.readbin2
    #
    filename_cal_rx_short = "/home/james/project/uhd/host/build/examples/data_backup/usrp_samples_loopback_short_0.dat"
    filename_cal_rx_open = "/home/james/project/uhd/host/build/examples/data_backup/usrp_samples_loopback_open_0.dat"
    filename_cal_rx_load = "/home/james/project/uhd/host/build/examples/data_backup/usrp_samples_loopback_load_0.dat"
    v_rx_short = readbin2(filename_cal_rx_short, N * Navg, offset=0)
    v_rx_open = readbin2(filename_cal_rx_open, N * Navg, offset=0)
    v_rx_load = readbin2(filename_cal_rx_load, N * Navg, offset=0)
    v_rx_short_avg = np.array(
        [sum(v_rx_short[i::N]) for i in range(len(v_rx_short) // Navg)]) / Navg  # Coherent average
    v_rx_open_avg = np.array(
        [sum(v_rx_open[i::N]) for i in range(len(v_rx_open) // Navg)]) / Navg  # Coherent average
    v_rx_load_avg = np.array(
        [sum(v_rx_load[i::N]) for i in range(len(v_rx_load) // Navg)]) / Navg  # Coherent average


    #
    filename_tx = "/home/james/project/uhd/host/build/examples/usrp_samples.dat"
    filename_meas_rx = "/home/james/project/uhd/host/build/examples/data_backup/usrp_samples_loopback_0.dat"#calibration/short/usrp_samples_loopback_short_1.dat" #/calibration/short
    v_tx = readbin2(filename_tx, N, offset=0)
    v_rx = readbin2(filename_meas_rx, N*Navg , offset=0)
    v_rx_avg = np.array([sum(v_rx[i::N]) for i in range(len(v_rx) // Navg)])/Navg  # Coherent average

    v_rx_avg_cal = main(v_rx= v_rx_avg, v_tx=v_tx, v_rx_short=v_rx_short_avg, v_rx_open=v_rx_open_avg, v_rx_load=v_rx_load_avg,N=N)

    #plt.plot(n, v_rx[0:N].real,'k-', n, v_rx_avg_cal.real, 'y-.')
    #plt.plot(n, v_rx[0:N].imag, 'k-', n, v_rx_avg_cal.imag, 'y-.')
    #plt.plot(n, rx_avg.real, 'k-')
    #plt.plot(n, v_rx.real[0:N], 'b-.', n, v_rx.imag[0:N], 'r-.')
    plt.plot(n,v_rx_avg_cal.real, 'k-.',n,v_rx_avg_cal.imag, 'y-.')
    #plt.legend(['Before Calibration', 'After Calibration'])
    plt.title('Time Domain Calibrated Signal')
    plt.xlabel('Sample number')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.show()