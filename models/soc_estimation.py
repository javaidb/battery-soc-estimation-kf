import numpy as np

def coulomb_counting(current, time, initial_soc, capacity):
    dt = np.diff(time)
    charge = np.cumsum(current[1:] * dt)
    soc = initial_soc - charge / (capacity * 3600)
    return np.clip(soc, 0, 1)