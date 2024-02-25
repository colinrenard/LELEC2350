import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Useful parameters
c = 3e8                     # Speed of light
wc = 2*np.pi*2e9            # cutoff pulsation of the waveguide

z_resolution = int(1e4)
w_resolution = int(1e2)
t_resolution = int(50)

endtime = 20/3e8
endpos = 20

z = np.linspace(0, endpos, z_resolution)
w = 2*np.pi*np.linspace(0.2e9, 4e9, w_resolution)    
t = np.linspace(0, endtime, t_resolution)
bandwidth = 2e9

# Definition of the spectrum of the electric field
mu = 3e9*2*np.pi            # center of the band
sigma = 0.3e9*2*np.pi       # standard deviation
Ehat = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((w-mu)/sigma)**2)
Ehat /= max(Ehat)           # Normalize spectrum

# Computes the electric field in the non-dispersive case
def computeE_nonDispersive(z, w, Ehat, t):
    """
    Computes the electric field in the non-dispersive case
    
    Input:
        z : Positions array
        w : Frequency array
        Ehat : Fourier transform of the electric field (array)
        t : time at which the field will be evaluated (float)

    Output:
        E : electric field at positions z evaluated at time t, normalized
    """
    E = np.zeros_like(z)
    for index, pos in enumerate(z):
        E[index] = np.trapz(np.real(Ehat * np.exp(1j*w*(t-pos/c))), w)
    # E = E/np.pi
    return E/5e9

def computeE_Dispersive(z, w, Ehat, t, wc):
    """
    Computes the electric field in a waveguide
    
    Input:
        z : Positions array
        w : Frequency array
        Ehat : Fourier transform of the electric field (array)
        t : time at which the field will be evaluated (float)
        wc : cutoff pulsation (float) [rad/s] 

    Output:
        E : electric field at positions z evaluated at time t, normalized
    """
    E = np.zeros_like(z)
    k = -1j/c * np.sqrt(1j*(w+wc)) * np.sqrt(1j*(w-wc))
    for index, pos in enumerate(z):    
        E[index] = np.trapz(np.real(Ehat * np.exp(1j*(w*t-k*pos))), w)
    E = E/np.pi
    return E/1e9

# Animation
fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)

def update(frame):
    line1.set_data([], [])
    line2.set_data([], [])
    E1 = computeE_nonDispersive(z, w, Ehat, t[frame])
    E2 = computeE_Dispersive(z, w, Ehat, t[frame], wc)
    line1.set_data(z, E1)
    line2.set_data(z, E2)
    ax.set_title(f'Time = {t[frame]*1e9:.2f} ns')
    ax.set_xlim(0, 20) 
    ax.set_ylim(-1.2, 1.2)

    return line1, line2

# Set axis labels
ax.set_xlabel('Position (z)')
ax.set_ylabel('Electric Field')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), interval=1)
plt.show()