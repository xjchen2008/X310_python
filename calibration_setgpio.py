# This script sets up the Numato 8 channel usb-gpio which controls Mini-circuit ZSDR-425 TTL RF switch
# https://github.com/numato/samplecode/tree/master/RelayAndGPIOModules/USBRelayAndGPIOModules/python/usbgpio8
# Author: Xingjian Chen
# Date: 20220205
import sys
import serial


def gpio_read():
    '''if (len(sys.argv) < 2):
        print
        "Usage: gpioread.py <PORT> <GPIONUM>\nEg: gpioread.py COM1 0"
        sys.exit(0)
    else:
        portName = sys.argv[1];
        gpioNum = sys.argv[2];
    '''
    # Open port for communication
    portName = "/dev/ttyACM0"
    serPort = serial.Serial(portName, 19200, timeout=1)

    # Send "gpio readall" command
    command = "gpio readall\r" # \r is a must at the end
    serPort.write(command.encode())
    response = serPort.read(25)
    print('Read from USB GPIO: ',  response[-1-4:-1-2])
    serPort.close()


def gpio_write(term='load'):
    # set the usb gpio port voltage.
    portName = "/dev/ttyACM0"
    # Open port for communication
    serPort = serial.Serial(portName, 19200, timeout=1)
    if term == 'short':  # Port RF1
        channel_num = "81"
    elif term == 'load':  # Port RF3
        channel_num = "82"
    elif term == 'open': # Port RF2
        channel_num = "83"
    elif term == 'antenna': # Port RF4
        channel_num = "80"
    command = "gpio writeall " + channel_num + "\r"
    # Send the command
    serPort.write(command.encode())

    # Close the port
    print("Write to USB GPIO: switched to " + channel_num)
    serPort.close()


def main(term='load'):
    gpio_write(term)
    gpio_read()


if __name__ == '__main__':
    main(term='short')

