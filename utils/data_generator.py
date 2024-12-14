import pybamm
import pandas as pd
import numpy as np
import os

class PyBAMMDataset():
    def __init__(
            self,
            capacity_Ah, 
            initial_soc_percent, 
            cc_demand_A, 
            evaluation_time_step, 
            num_points
        ):
        self.capacity_Ah = capacity_Ah # Ah

        self.initial_soc_percent = initial_soc_percent
        self.cc_demand_A = cc_demand_A

        self.evaluation_time_step = evaluation_time_step # s
        self.num_points = num_points # datapoints

        self.data = type('DataContainer', (object,), {})()


    def update_sim_vector(self, **kwargs):
        # self.dataset_vector = {key: value for key, value in kwargs.items()}

        for key, value in kwargs.items():
            setattr(self.data, key, value)
        # self.data.update(kwargs)
        return

    def change_pybamm_param_capacity(self, parameter_values, capacity_Ah_new):
        default_capacity = parameter_values["Nominal cell capacity [A.h]"]
        old_height = parameter_values['Electrode height [m]']
        old_width = parameter_values['Electrode width [m]']
        capacity_scaling_factor = capacity_Ah_new/default_capacity
        parameter_values["Nominal cell capacity [A.h]"] = capacity_Ah_new
        parameter_values['Electrode height [m]'] = capacity_scaling_factor*old_height  # Set new height
        parameter_values['Electrode width [m]'] = capacity_scaling_factor*old_width    # Set new width
        return parameter_values

    def __find_parent_directory(self):
        # Traverse up dir tree to find 'battery-state-estimation'
        current_dir = os.getcwd()
        while True:
            if os.path.basename(current_dir) == 'battery-state-estimation':
                return current_dir
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root dir
                raise FileNotFoundError("Could not find parent 'battery-state-estimation' directory.")
            current_dir = parent_dir

    def generate_ocv_soc_lookup_table(self):

        file_name = f'ocv_soc_mapping_pybamm_{self.capacity_Ah}Ah.pkl'
        
        parent_dir = self.__find_parent_directory()
        datasets_path = os.path.join(parent_dir, 'datasets')
        if not os.path.exists(datasets_path):
            os.mkdir(datasets_path)
            
        save_path = os.path.join(datasets_path, file_name)
        if not os.path.exists(save_path):
            # Define the SOC range from 0 to 1 (0% to 100%)
            socs = np.linspace(0, 1, 101)  # 101 points for 1% resolution
        
            charge_voltages = np.zeros(101)
            discharge_voltages = np.zeros(101)
        
            # Load the model (for example, using the 'DFN' model)
            model_init = pybamm.lithium_ion.DFN()  # DFN: Doyle-Fuller-Newman model
            parameter_values = pybamm.ParameterValues("Chen2020")  # Example parameter set
        
            parameter_values = self.change_pybamm_param_capacity(parameter_values, self.capacity_Ah)
            
            
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
            
            dchg_v = dchg_sim.solution["Battery open-circuit voltage [V]"].entries[::-1]
            num_ocv_points_dchg = len(chg_v)
            soc_values = np.linspace(0, 100, num_ocv_points_dchg)
            soc_resolution = np.arange(0, 101, 1)  # From 0% to 100%
            ocv_interpolated_dchg = np.interp(soc_resolution, soc_values, dchg_v)
        
            df = pd.DataFrame({
                'soc': soc_resolution,
                'voltage_chg': ocv_interpolated_chg,
                'voltage_dchg': ocv_interpolated_dchg
            })
            df.to_pickle(save_path)
            print(f"OCV-SOC mapping saved to '{save_path}'")
        else:
            df = pd.read_pickle(save_path)
            print(f"OCV-SOC mapping located/retrieved from '{save_path}'")
            soc_resolution = df.soc.values
            ocv_interpolated_chg = df.voltage_chg.values
            ocv_interpolated_dchg = df.voltage_dchg.values
        
        def interpolate_chg(value):
            """Interpolates the y value for a given x value."""
            if value < soc_resolution[0] or value > soc_resolution[-1]:
                raise ValueError("Value is outside the interpolation range.")
            return np.interp(value, soc_resolution, ocv_interpolated_chg)
        
        def interpolate_dchg(value):
            """Interpolates the y value for a given x value."""
            if value < soc_resolution[0] or value > soc_resolution[-1]:
                raise ValueError("Value is outside the interpolation range.")
            return np.interp(value, soc_resolution, ocv_interpolated_dchg)
        
        return interpolate_chg, interpolate_dchg

    def generate_sim_single_phase(self):
        """
        Generate simulated battery data including voltage, current, temperature, and time.

        Parameters:
        num_points (int): Number of data points to generate.
        current (float): The constant current to simulate in Amperes.

        Returns:
        tuple: Arrays of voltage, current, temperature, and time.
        """

        time = np.arange(0, self.num_points * self.evaluation_time_step, self.evaluation_time_step)

        model = pybamm.lithium_ion.SPMe()  # Single Particle Model with electrolyte
        parameter_values = pybamm.ParameterValues("Chen2020")
        
        default_capacity = parameter_values["Nominal cell capacity [A.h]"]
        old_height = parameter_values['Electrode height [m]']
        old_width = parameter_values['Electrode width [m]']
        capacity_scaling_factor = self.capacity_Ah/default_capacity
        parameter_values["Nominal cell capacity [A.h]"] = self.capacity_Ah
        parameter_values['Electrode height [m]'] = capacity_scaling_factor*old_height  # Set new height
        parameter_values['Electrode width [m]'] = capacity_scaling_factor*old_width    # Set new width
        
        parameter_values['Current function [A]'] =  self.cc_demand_A
        
        simulation = pybamm.Simulation(model, parameter_values=parameter_values)

        simulation.solve(t_eval=time, initial_soc = self.initial_soc_percent/100)
        self.sim = simulation

        entries_of_interest  = [
            "Time [s]",
            "Terminal voltage [V]",
            "Current [A]",
            "Battery open-circuit voltage [V]"
        ]
        extracted_data = self.extract_entries_from_sim(entries_of_interest)
        self.update_sim_vector(
            time=extracted_data[0],
            voltage=extracted_data[1],
            current=extracted_data[2],
            ocv=extracted_data[3]
        )
        return simulation


    def extract_entries_from_sim(
        self,
        list_of_entries,
        sim=None):
        
        def is_3d_entry(entry_arr):
            if len(entry_arr.shape) == 2:
                return True
        
        if sim is None:
            sim = self.sim
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


    # def simply_generate_standard_arrays(
    #     self,
    #     current, 
    #     num_points, 
    #     dt=60,
    #     capacity_Ah=3,
    #     initial_soc_percent=100):
    #     """
    #     Generate simulated battery data including voltage, current, temperature, and time.

    #     Parameters:
    #     num_points (int): Number of data points to generate.
    #     current (float): The constant current to simulate in Amperes.

    #     Returns:
    #     tuple: Arrays of voltage, current, temperature, and time.
    #     """
        
    #     simulation = self.generate_sim_single_phase(
    #         current, 
    #         num_points, 
    #         dt=60,
    #         capacity_Ah=3,
    #         initial_soc_percent=100
    #     )

    #     simulation.solve(t_eval=time, initial_soc = initial_soc_percent/100)

    #     time, voltage, current, temperature = self.extract_entries_from_sim(
    #         simulation,
    #         ["Time (s)", "Terminal voltage [V]", "Current [A]", "Cell temperature [K]"]
    #     )
        
    #     return time, voltage, current, temperature

class SimpleDataset():
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def generate_battery_data(self):
        time = np.linspace(0, 3600, self.num_samples)
        voltage = 3.7 + 0.3 * np.sin(time / 1000)
        current = 2 + 0.5 * np.random.randn(self.num_samples)
        temperature = 25 + 5 * np.random.randn(self.num_samples)
        
        return time, voltage, current, temperature