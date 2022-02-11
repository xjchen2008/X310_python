import numpy as np
import timeit
import os
import struct
from numpy import fft
import time
import matplotlib.pyplot as plt
import functions as fn
plt.rc('font', size=15)

def save_tx_canc(filename, N, y_hat):
    # save y_hat
    y_cxnew = np.around(32767 * y_hat)  # numpy.multiply(y_cx,win) 6.9
    yw = np.zeros(2 * N)
    for i in range(0, N):
        yw[2 * i + 1] = np.imag(y_cxnew[i])  # tx signal
        yw[2 * i] = np.real(y_cxnew[i])  # tx signal
    #yw = np.append(yw, yw[-2])  # Manually add one more point at the end of the 4096 points pulse to match the E312 setup
    yw = np.append(yw, yw[-2])
    yw = np.int16(yw)  # E312 setting --type short
    data = open(filename, 'wb')
    data.write(yw)
    data.close()


def plot(N, y, yc, x_tx):
    n = np.linspace(1, N, N)
    '''
    plt.figure()
    plt.plot(y.real,y.imag)
    plt.title('imbalance constellation')
    plt.figure()
    plt.plot(yc.real,yc.imag)
    plt.title('corrected constellation')
    '''

    plt.figure()
    plt.plot(n, y.real, n, y.imag)
    plt.title('RX Signal: IQ-imbalance and delay')
    plt.figure()

    plt.plot(n, yc.real, n, yc.imag)
    plt.title('Corrected Signal in PDP Simulation')
    plt.figure()
    plt.plot(n, x_tx.real, n, x_tx.imag)
    plt.title('Updated TX Signal')
    plt.show()


def main():
    start = timeit.default_timer()
    # initialization
    N = 1000#4096
    read_bin = fn.ReadBin()
    x = read_bin.readbin2("/home/james/project/uhd/host/build/examples/mypython/usrp_samples_0.dat", count=N, offset=0)
    y = np.zeros(N)
    # averaging the last pulses
    Navg = 2

    for idx in range(1, Navg):
        y0 = np.squeeze(np.reshape(read_bin.last_pulse(filename="usrp_samples_loopback.dat", nsamples=N, rubish=0, channel_num=1), [N, 1]), axis= 1)
        y = y+y0
    y = y/(1.0*Navg)
    y = 15*y  # tunable digital scaling for setting the generated x_tx less than 1

    ########################
    X = np.fft.fft(x)
    Y = np.fft.fft(y)
    H = np.divide(Y, X)
    H_inv = 1/H

    print(np.fft.ifft(H_inv).real.mean())
    print(np.fft.ifft(H_inv).imag.mean())
    X_tx = np.multiply(H_inv, X)
    X_tx[0] = 0 # setting the bin zero(DC component to zero). It is not zero due to noise and randomness.
    x_tx = np.fft.ifft(X_tx)
    x_tx_normalize = x_tx / max(abs(x_tx))
    X_tx_normalize = np.fft.fft(x_tx_normalize)
    save_tx_canc('../usrp_samples.dat', N, np.array(x_tx_normalize))  # add FGPA tx delay inside this function
    Yc = np.multiply(H, X_tx_normalize)
    yc = np.fft.ifft(Yc)
    plot(N, y, yc, x_tx_normalize) # uncomment this line for plotting

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == "__main__":  # Change the following code into the c++
    main()