import pybamm
import pandas as pd
import numpy as np

def change_pybamm_param_capacity(parameter_values, capacity_Ah_new):
    default_capacity = parameter_values["Nominal cell capacity [A.h]"]
    old_height = parameter_values['Electrode height [m]']
    old_width = parameter_values['Electrode width [m]']
    capacity_scaling_factor = capacity_Ah_new/default_capacity
    parameter_values["Nominal cell capacity [A.h]"] = capacity_Ah_new
    parameter_values['Electrode height [m]'] = capacity_scaling_factor*old_height  # Set new height
    parameter_values['Electrode width [m]'] = capacity_scaling_factor*old_width    # Set new width
    return parameter_values

def generate_ocv_soc_lookup_table(capacity_Ah, c_rate=1/100):
    # Define the SOC range from 0 to 1 (0% to 100%)
    socs = np.linspace(0, 1, 101)  # 101 points for 1% resolution

    charge_voltages = np.zeros(101)
    discharge_voltages = np.zeros(101)

    # Load the model (for example, using the 'DFN' model)
    model_init = pybamm.lithium_ion.DFN()  # DFN: Doyle-Fuller-Newman model
    parameter_values = pybamm.ParameterValues("Chen2020")  # Example parameter set

    parameter_values = change_pybamm_param_capacity(parameter_values, capacity_Ah)
    
    
    experiment_pre = pybamm.Experiment(
        [
            pybamm.step.string("Discharge at C/100 until 2.5V"),
            pybamm.step.string("Rest for 5 hour"),
        ]
    )
    pre_sim = pybamm.Simulation(model_init, experiment=experiment_pre, parameter_values=parameter_values)
    print("Solving sim (PRE)...")
    pre_sim.solve()
    
    
    experiment_chg = pybamm.Experiment(
        [pybamm.step.string("Charge at C/100 until 4.2 V")]
    )
    model_chg = model_init.set_initial_conditions_from(pre_sim.solution, inplace=True)
    chg_sim = pybamm.Simulation(model_chg, experiment=experiment_chg, parameter_values=parameter_values)
    print("Solving sim (CHG)...")
    chg_sim.solve()
    
    
    experiment_rest = pybamm.Experiment(
        [pybamm.step.string("Rest for 5 hour")]
    )
    model_rest = model_init.set_initial_conditions_from(chg_sim.solution, inplace=False)
    rest_sim = pybamm.Simulation(model_rest, experiment=experiment_rest, parameter_values=parameter_values)
    print("Solving sim (REST)...")
    rest_sim.solve()
    
    
    experiment_dchg = pybamm.Experiment(
        [pybamm.step.string("Discharge at C/100 until 2.5 V")]
    )
    model_dchg = model_init.set_initial_conditions_from(rest_sim.solution, inplace=False)
    dchg_sim = pybamm.Simulation(model_dchg, experiment=experiment_dchg, parameter_values=parameter_values)
    print("Solving sim (DCHG)...")
    dchg_sim.solve()
    dchg_sim_sol = dchg_sim.solution
    
    chg_v = chg_sim.solution["Battery open-circuit voltage [V]"].entries
    num_ocv_points_chg = len(chg_v)
    soc_values = np.linspace(0, 100, num_ocv_points_chg)
    soc_resolution = np.arange(0, 101, 1)  # From 0% to 100%
    ocv_interpolated_chg = np.interp(soc_resolution, soc_values, chg_v)
    
    dchg_v = dchg_sim.solution["Battery open-circuit voltage [V]"].entries
    num_ocv_points_dchg = len(chg_v)
    soc_values = np.linspace(0, 100, num_ocv_points_dchg)
    soc_resolution = np.arange(0, 101, 1)  # From 0% to 100%
    ocv_interpolated_dchg = np.interp(soc_resolution, soc_values, dchg_v)
    
    def interpolate_chg(value):
        """Interpolates the y value for a given x value."""
        if value < soc_values[0] or value > soc_values[-1]:
            raise ValueError("Value is outside the interpolation range.")
        return np.interp(value, soc_resolution, ocv_interpolated_chg)
    
    def interpolate_dchg(value):
        """Interpolates the y value for a given x value."""
        if value < soc_values[0] or value > soc_values[-1]:
            raise ValueError("Value is outside the interpolation range.")
        return np.interp(value, soc_resolution, ocv_interpolated_dchg)
    
    return interpolate_chg, interpolate_dchg

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