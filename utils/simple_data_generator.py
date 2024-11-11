import numpy as np

def generate_battery_data(num_samples):
    time = np.linspace(0, 3600, num_samples)
    voltage = 3.7 + 0.3 * np.sin(time / 1000)
    current = 2 + 0.5 * np.random.randn(num_samples)
    temperature = 25 + 5 * np.random.randn(num_samples)
    
    return time, voltage, current, temperature