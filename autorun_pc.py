import wavetable_mixer_4096
from matplotlib import pyplot as plt
import numpy as np
import functions as fn
from numpy.fft import fftshift
import os
from natsort import natsorted, ns
import calibration_1port as cal

#plt.rc('font', size=15)
path = '/home/james/project/uhd/host/build/examples/data_backup'
os.chdir(path)
path_meas = path+'/m11_ant/measurement/'
path_short = path+'/m11_ant/calibration/short/'
path_open = path+'/m11_ant/calibration/open/'
path_load = path+'/m11_ant/calibration/load/'

filenames = next(os.walk(path_meas), (None, None, []))[2]  # [] if no file
filenames_short = next(os.walk(path_short), (None, None, []))[2]
filenames_open = next(os.walk(path_open), (None, None, []))[2]
filenames_load = next(os.walk(path_load), (None, None, []))[2]
filenames.sort()

filenames = natsorted(filenames, key=lambda y: y.lower())#sort alphanumeric in ascending order
filenames_short = natsorted(filenames_short, key=lambda y: y.lower())
filenames_open = natsorted(filenames_open, key=lambda y: y.lower())
filenames_load = natsorted(filenames_load, key=lambda y: y.lower())

print('The number of files=', len(filenames))
N = 1000#1000  # 40000#1120000*2
Navg = 1 #50#4000#40#0*10#1 # 400
M = 1 #100
c = 3e8
fs = 200e6
# Fp0 = 500e3/10*2
rubish = 2000  # 50*N #10000 #1000 #8000*50#*1000# *9#256*8
rubish_tx_pulse = 1000
rubish_tx_FMCW = 0
read_bin = fn.ReadBin()
tx = read_bin.readbin2("/home/james/project/uhd/host/build/examples/usrp_samples.dat", N, rubish_tx_FMCW)
#tx = read_bin.readbin2("/home/james/project/uhd/host/build/examples/mypython/usrp_samples_0.dat", N, 0)
freq = np.fft.fftfreq(N, d=1. / fs)
offset_cal = 0  # a calibration distance due to Ettus delay and cable delay
# distance = c * freq /(2*M*np.power(Fp0, 2)) *4 - offset_cal
Rmax1 = N * c / (2 * fs)  # 1/del_F *c/2
distance = np.linspace(0, Rmax1, N) - offset_cal  # FMCW PC=Matched Filter radar
bw = 20e6
Tp = N* 1/fs
distance = c / 2 * freq / (bw / Tp) - offset_cal # FMCW PC = Stretch method

#################################
# Data processing for measurement
#################################
def generate_pc_file(plot=False):
    read_bin = fn.ReadBin()
    readbin2 = read_bin.readbin2
    filename_tx = "/home/james/project/uhd/host/build/examples/usrp_samples.dat"
    v_tx_cal = readbin2(filename_tx, N, offset=0)

    for idx in range(len(filenames)):
        print(filenames[idx])
        filename_cal_rx_short = path_short + filenames_short[idx]
        filename_cal_rx_open = path_open + filenames_open[idx]
        filename_cal_rx_load = path_load + filenames_load[idx]
        filename_meas = path_meas + filenames[idx]
        if idx < len(filenames)-1:
            filename_cal_rx_short_offset = path_short + filenames_short[idx+1]
            filename_cal_rx_load_offset = path_load + filenames_load[idx + 1]
            filename_cal_rx_meas_offset = path_meas + filenames[idx + 1]
        if idx == len(filenames)-1:
            filename_cal_rx_short_offset = path_short + filenames_short[idx-1]
            filename_cal_rx_load_offset = path_load + filenames_load[idx - 1]
            filename_cal_rx_meas_offset = path_meas + filenames[idx - 1]
        v_rx_short = readbin2(filename_cal_rx_short, N, offset=0)
        v_rx_open = readbin2(filename_cal_rx_open, N, offset=0)
        v_rx_load = readbin2(filename_cal_rx_load, N, offset=0)
        v_rx = read_bin.readbin2(filename_cal_rx_meas_offset, N * Navg, offset=rubish)  # Chen: change the meas
        # filename here!
        v_rx_avg = np.array([sum(v_rx[i::N]) for i in range(len(v_rx) // Navg)])/Navg  # Coherent average
        v_rx_avg_cal = cal.main(v_rx=v_rx_avg, v_tx=v_tx_cal, v_rx_short=v_rx_short,
                          v_rx_open=v_rx_open, v_rx_load=v_rx_load,N=N) # calibrate the signal at rx end
        # pc calculation 1
        pc1 = fn.PulseCompr(rx=v_rx_avg_cal, tx=tx, win=1, unit='linear') # Chen: change rx signal here for test
        pc1_db = (20 * np.log10(abs(pc1)))
        # pc1_db_normalized = pc1_db - pc1_db.max()
        savefile = path + '/pc/pc_' + filenames[idx][-1 - 14:-1 - 3]
        np.save(savefile, pc1)
        if plot == True:

            # plt.figure()
            plt.plot(fftshift(distance), fftshift(pc1_db), )  # Matched Filter PC
            plt.xlabel('Distance [m]')
            #plt.plot(pc1_db, 'b*-')
            #plt.xlabel('Sample Number')
            plt.ylabel('Magnitude [dB]')
            plt.grid()
            #plt.xlim([-20, 1000])
            #plt.ylim([-50, 40])
            plt.title('Pulse Compression')
            '''
            plt.plot(v_rx_avg_cal.real)
            plt.plot(v_rx_avg_cal.imag)
            plt.xlabel('Sample Number')
            plt.title('Time Domain Rx Signal')
            '''
        else:
            pass
    plt.show()


def time_series():
    # filenamepc = next(os.walk(path + '/pc/NoCar_20220101'), (None, None, []))[2]  # [] if no file
    filenamepc = next(os.walk(path + '/pc/'), (None, None, []))[2]  # [] if no file
    filenamepc.sort
    pc = np.array([])
    for idx in range(len(filenamepc)):
        # pc0=np.load(path+'/pc/NoCar_20220101/'+filenamepc[idx])
        pc0 = np.load(path + '/pc/' + filenamepc[idx])
        #pc0 = pc0 * np.exp(-1j * (np.angle(
        #    pc0[0]) - 2 * np.pi))  # Normalize phase according to a pc peak which is the strongest echo.
        pc0_db = (20 * np.log10(abs(pc0)))
        pc0_db_normalized = pc0_db - pc0_db.max()
        pc = np.append(pc, pc0, 0)
        plt.plot(fftshift(freq)/1e6, fftshift(pc0_db)[0:1000])
    pc = pc.reshape((int(len(pc) / (N))), N)
    plt.xlabel('Frequency [MHz]')
    #plt.xlabel('Distance [m]')
    #plt.xlabel('Sample Number')
    plt.ylabel('Magnitude [dB]')
    plt.title('Pulse Compression')
    plt.grid()
    plt.figure()
    # loc = [73,272, 472, 673, 874]   #110 #171
    # loc = [73, 272, 472,874]
    loc = [0]
    loc_str = [str(integer) for integer in loc]

    pc_samp_timeseries = np.abs(pc[:, loc])
    pc_db_samp_timeseries = 20 * np.log10(np.abs(pc[:, loc]))
    var = np.var(pc_samp_timeseries, axis=0)
    print(var)
    plt.plot(20 * np.log10(np.abs(pc[:, loc])), '*-')
    plt.xlabel('Time Series Number')
    plt.ylabel('Magnitude [dB]')
    plt.title('Time Series of PC power at sample number ' + loc.__str__())
    plt.grid()
    plt.legend(loc_str)

    plt.figure()
    plt.plot((np.angle(pc[:, loc])), '*-')
    plt.xlabel('Time Series Number')
    plt.ylabel('Angle [rad]')
    plt.title('Time Series of PC phase at sample number ' + loc.__str__())
    plt.grid()
    plt.show()

    return var


######################
# Simulation functions
######################
def sim_precision(n_std=1, N_average=10, N_measure=100):
    # This is to simulate the precision of the radar.
    # Calculate the PC var and mean in different measurements with certain noise level.
    var = np.array([])
    mu = np.array([])
    pc_series_loc_mag_linear = np.array([])
    pc_series_loc_mag_db = np.array([])
    pc_series_loc_angle = np.array([])
    N_measure = N_measure
    N_average = N_average
    mean = 0  # mean of the Normal distribution
    std = n_std  # standard deviation of the Normal distribution
    num_samples = N
    A = 1
    tx = wavetable_mixer_4096.y_cx
    np.random.seed(10)
    for m in range(N_measure):
        # In one measurement, take N_average coherent average for the Pulse Compression, with Additive white Gaussian
        # noise (AWGN)
        rx = np.zeros(num_samples)  # initial value and re-zeros for every measurement init value
        for i in range(N_average):
            noise = np.random.normal(mean, std, size=num_samples) + 1j * np.random.normal(mean, std, size=num_samples)
            f_tree = 0.01  # lower value means slower variation or longer period.
            rx0 = .1 + 0.001 * np.sin(2 * f_tree * np.pi * np.linspace(0, N, N)) + noise  # A*tx + noise
            rx = rx + rx0  # sum
        rx_avg = rx / N_average  # avg
        pc_avg = rx_avg #fn.PulseCompr(rx=rx_avg, tx=tx, win=1, unit='linear')
        # pc = pc + pc
        # pc_avg = pc / N_average
        pc_avg_db = 20 * np.log10(np.abs(rx_avg))
        # Take the sample of the peak value of PC
        loc = 0  # peak value location
        pc_series_loc_mag_linear = np.append(pc_series_loc_mag_linear, pc_avg[loc]) # rx_avg[loc] pc_avg[loc]
        # pc_series_loc_mag_linear = np.append(pc_series_loc_mag_linear, abs(rx_avg.mean()))
        pc_series_loc_mag_db = np.append(pc_series_loc_mag_db, pc_avg_db[loc])
        pc_series_loc_angle = np.append(pc_series_loc_angle, np.angle(rx_avg[loc]))

    var_1 = np.var(
        pc_series_loc_mag_linear)  # Var{|X|}; The variance of a value which should be in linear scale. Generate one variance
    var_2 = np.var(
        pc_series_loc_mag_db)  # Var{|X|dB}; A variance after transform in dB scale from linear scale. Generate one variance

    std_1 = 2*np.std(
        pc_series_loc_mag_linear)  # Var{|X|}; The variance of a value which should be in linear scale. Generate one variance
    std_2 = 2*np.std(
        pc_series_loc_mag_db)  # Var{|X|dB}; A variance after transform in dB scale from linear scale. Generate one variance

    # for N_measure measurements of peak pc value
    mu = np.mean(pc_series_loc_mag_linear)  # mean of the sample[loc] in PC
    '''
    plt.figure()
    plt.plot(pc_series_loc_angle, '*-')
    plt.xlabel('Time Series Number')
    plt.ylabel('Angle [rad]')
    plt.title('Time Series of PC phase at sample number ' + loc.__str__())
    '''
    '''
    plt.figure()
    plt.plot(pc_series_loc_mag, '*-')
    plt.xlabel('Measurement Series Number')
    plt.ylabel('Amplitude [dB]')
    plt.title('Time Series of PC Amplitude at sample number ' + loc.__str__())
    plt.show()
    '''
    return std_1, std_2, pc_avg_db


def run_sim():
    N_measure = 10
    pc_sa_var = np.array([])
    n_average_range = [10, 100, 1000]
    n_std_range = [1, 1e-1, 1e-2, 1e-3]
    pc_sa_var_all = np.zeros(len(n_average_range))

    for n_std in n_std_range:
        plt.figure()
        for N_average in n_average_range:
            print('n_std=', n_std, 'N_average=', N_average)
            pc_sa_var0, pc_sa_var1, pc_avg_db = sim_precision(n_std=n_std, N_average=N_average, N_measure=N_measure)
            pc_sa_var0_db = 10 * np.log10(pc_sa_var0)  # convert variance from linear to dB scale; Var is std^2,
            # so 10log()
            pc_sa_var = np.append(pc_sa_var, pc_sa_var1) #np.append(pc_sa_var, pc_sa_var0_db)
            plt.plot(pc_avg_db)
            snr = 20 * np.log10((0.1 + 0.001) / n_std) # for plot
            snr_str = "SNR before averaging = {snr:.2f}".format(snr = snr) # set the display precision to 2 digits
            plt.title('AWGN Noise~$N(0,\sigma)$, $\sigma$ = ' + n_std.__str__()
                      + '\n Signal Amplitude y = $0.1 + 0.001sin(2f_{tree}*x)+noise$'
                      + '\n'+snr_str +'dB')# set snr display precision to 3 digits
            plt.ylabel('Magnitude [dB]')
            plt.xlabel('Sample Number $x$')
        for integer in pc_sa_var:
            legned = str(integer)
            N_average
        plt.legend([str(integer) for integer in pc_sa_var]) # pc_sa_var , n_average_range
        pc_sa_var_all = np.row_stack((pc_sa_var_all, pc_sa_var))
        pc_sa_var = np.array([])
    n_std_range.insert(0, 0)
    n_std_range_log = 1 * np.log10(n_std_range)
    plt.figure()
    plt.plot(n_std_range_log, pc_sa_var_all[:, 0], '-*k')
    plt.plot(n_std_range_log, pc_sa_var_all[:, 1], '-ob')
    plt.plot(n_std_range_log, pc_sa_var_all[:, 2], '-^r')
    # plt.plot(n_std_range_log, pc_sa_var_all[:, 3], '-vy')
    plt.title('Variance V.S. AWGN std and Average Times')
    plt.xlabel('Log scale noise standard deviation')
    plt.ylabel('Variance of the dB peak power sample')
    plt.legend(n_average_range)
    plt.grid('on')
    plt.show()


if __name__ == '__main__':
    #run_sim()

    generate_pc_file(plot=True)
    var= time_series()

    print('Done')