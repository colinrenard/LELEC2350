import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Nt = 200; Nz = int(1e4); Nf = 150       # Resolution in time - space - frequency of the simulation 
z = np.linspace(0,10,Nz)                # z-axis [m]
fc = 3e9                                # Central frequency [Hz]
df = 1e9                                # Deviation frequency [Hz]
f = np.linspace(fc-df,fc+df,Nf)         # Frequency axis [Hz]
B = (fc+df) - (fc-df)                   # Bandwidth [Hz]
t = np.linspace(0,100*(1/fc),Nt)        # Time axis [s]
w,wc = 2*np.pi*f, 2*np.pi*fc            # Pulsation axis & central pulsation
sigma_w = 2*np.pi*0.2*df                # Standard deviation of pulsations

A = (1/(np.sqrt(2*np.pi)*sigma_w)) * np.exp(-(w-wc)**2/(2*(sigma_w**2)))        # Amplitude distribution over frequences (Gaussian)
# A = A*np.sqrt(4*sigma_w*(np.pi**(3/2)))                                       # Normalized

mu = 4*np.pi*1e-7                       # Permeability [H/m]
epsilon = 8.85*1e-12                    # Permittivity [F/m]
c = 1/(np.sqrt(mu*epsilon))             # Speed of light [m/s]

k = w/c # Wavenumbers
def compute_E_non_dispersive(z,t):
    E = np.zeros(len(z))
    for i in range(len(z)):
        E[i] = 2 * B *  np.real(np.sum(A*np.exp(-1j*k*z[i])*np.exp(1j*w*t))) # (1/pi) dw = 2 df
    return E

E_non_dispersive = np.zeros((Nt,Nz))
for i in range(Nt):    E_non_dispersive[i] = compute_E_non_dispersive(z,t[i])


wc_disp = 2*np.pi * 2e9                                             # Cutoff frequency of the waveguide
k_disp = (1/c) * np.sqrt((w+wc_disp)) * np.sqrt((w-wc_disp))        # Wavenumver (f) 
def compute_E_dispersive(z,t):
    # Computes s for an array z at a particular time t
    E = np.zeros(len(z))
    for i in range(len(z)):
        E[i] =  2 * B * np.real(np.sum(A*np.exp(-1j*k_disp*z[i])*np.exp(1j*w*t)))
    return E

E_dispersive = np.zeros((Nt,Nz))
for i in range(Nt):   E_dispersive[i] = compute_E_dispersive(z,t[i])

fig, ax = plt.subplots()
ax.set_xlabel("z [m]")
ax.set_ylabel("Electric Field [V/m]")
ax.set_title("Propagation in a dispersive and in a non-dispersive medium")
ax.grid()

line, = ax.plot(z, E_non_dispersive[0], label="Non dispersive medium")
line2, = ax.plot(z, E_dispersive[0], label="Dispersive medium (WG)")
ax.legend()

def anim(i):       
    #line.set_ydata(compute_s(z,t[i]))  # update the data
    line.set_ydata(E_non_dispersive[i])
    line2.set_ydata(E_dispersive[i])
    return line,

ani = animation.FuncAnimation(fig, anim, frames=len(t),interval=50, repeat=True ,blit=False)
plt.show()