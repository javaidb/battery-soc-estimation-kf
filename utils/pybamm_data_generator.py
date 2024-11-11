import pybamm
import pandas as pd
import numpy as np

def generate_sim_single_phase(
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
    
    parameter_values['Current function [A]'] =  current
    
    simulation = pybamm.Simulation(model, parameter_values=parameter_values)

    simulation.solve(t_eval=time, initial_soc = initial_soc_percent/100)
    
    return simulation


def extract_entries_from_sim(
    sim,
    list_of_entries):
    
    def is_3d_entry(entry_arr):
        if len(entry_arr.shape) == 2:
            return True
    
    all_entry_arrs = []
    for entry_name in list_of_entries:
        entries = sim.solution[entry_name].entries
        if is_3d_entry(entries):
            entries = np.array([np.mean(entries[:, col]) for col in range(entries.shape[1])])
        all_entry_arrs.append({
            'label': entry_name,
            'data': entries
        })

        # current_array = np.full_like(time, current)
    
    return tuple(all_entry_arrs)


def simply_generate_standard_arrays(
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
    
    simulation = generate_sim_single_phase(
        current, 
        num_points, 
        dt=60,
        capacity_Ah=3,
        initial_soc_percent=100
    )

    simulation.solve(t_eval=time, initial_soc = initial_soc_percent/100)

    time, voltage, current, temperature = extract_entries_from_sim(
        simulation,
        ["Time (s)", "Terminal voltage [V]", "Current [A]", "Cell temperature [K]"]
    )
    
    return time, voltage, current, temperature