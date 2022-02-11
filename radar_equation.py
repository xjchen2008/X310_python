# According to the equation in the web
# https://www.radartutorial.eu/01.basics/The%20Radar%20Range%20Equation.en.html
import numpy as np
import matplotlib.pyplot as plt

N = 1000
A = 1 #geometric antenna area [m²]
Ka = 0.5 #Anenna efficiency
sigma = 1
G = np.power(10, 50/10) # gain
Ps = np.power(10, 20/10) / 1000 # TX power 20dBm
R = np.linspace(1, 1000, N)
Pe = Ps*G*sigma*A*Ka/(np.power(4*np.pi,3)*np.power(R,4) ) # RX power
Pe_db = 10*np.log10(Pe)
pe_db_normalized = Pe_db- max(Pe_db)
plt.plot(R, pe_db_normalized)
plt.xlabel('Range [m]')
plt.ylabel('Received Power [dB]')
plt.title('Radar Equation for Received Power: $ \dfrac{P_sG \sigma AK_a}{(4\pi)^3R^4 }$')
plt.grid()
plt.show()
