import pybamm
import pandas as pd
import numpy as np

def simply_generate_battery_data(
    current, 
    num_points, 
    dt=60,
    capacity_Ah=3,
    initial_soc_percent=100):
    """
    Generate simulated battery data including voltage, current, temperature, and time.

    Parameters:
    num_points (int): Number of data points to generate.
    current (float): The constant current to simulate in Amperes.

    Returns:
    tuple: Arrays of voltage, current, temperature, and time.
    """

    time = np.arange(0, num_points * dt, dt)

    model = pybamm.lithium_ion.SPMe()  # Single Particle Model with electrolyte
    parameter_values = pybamm.ParameterValues("Chen2020")
    
    
    default_capacity = parameter_values["Nominal cell capacity [A.h]"]
    old_height = parameter_values['Electrode height [m]']
    old_width = parameter_values['Electrode width [m]']
    capacity_scaling_factor = capacity_Ah/default_capacity
    parameter_values["Nominal cell capacity [A.h]"] = capacity_Ah
    parameter_values['Electrode height [m]'] = capacity_scaling_factor*old_height  # Set new height
    parameter_values['Electrode width [m]'] = capacity_scaling_factor*old_width    # Set new width
    
    simulation = pybamm.Simulation(model, parameter_values=parameter_values)

    simulation.solve(t_eval=time, initial_soc = initial_soc_percent/100, inputs={"Current function [A]": current})

    voltage = simulation.solution["Terminal voltage [V]"].entries
    temperature_3d = simulation.solution["Cell temperature [K]"].entries
    temperature_2d = np.array([np.mean(temperature_3d[:, col]) for col in range(temperature_3d.shape[1])])

    current_array = np.full_like(time, current)

    min_length = min(len(voltage), len(temperature_2d), len(current_array), len(time))
    
    return (
        time[:min_length],
        voltage[:min_length],
        current_array[:min_length],
        temperature_2d[:min_length]
    )