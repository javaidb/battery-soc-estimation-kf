import numpy as np

def coulomb_counting(current, time, initial_soc_percent, capacity):
    initial_soc = initial_soc_percent/100
    dt = np.diff(time)
    charge = np.cumsum(current[1:] * dt)
    soc = initial_soc - charge / (capacity * 3600)
    return 100*np.clip(soc, 0, 1)