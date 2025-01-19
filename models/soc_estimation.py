import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

class StateEstimator():
    def __init__(self, Dataset):
        self.Dataset = Dataset
        self.data = Dataset.data
        self.SOCfromOCV_fn_chg, self.SOCfromOCV_fn_dchg = Dataset.generate_ocv_soc_lookup_table()

    def coulomb_counting(self):
        initial_soc = self.Dataset.initial_soc_percent/100
        dt = np.diff(self.data.time["data"])
        accumulated_charge = np.cumsum(self.data.current["data"][1:] * dt)
        soc = initial_soc - accumulated_charge / (self.Dataset.capacity_Ah * 3600)
        soc_with_initial = np.insert(soc, 0, initial_soc)
        return {
            "label": "Coulomb Counted SOC (%)",
            "data": np.clip(soc_with_initial, 0, 1)
        }

    def __determine_hysteresis(self, current_arr):
        if np.mean(current_arr) < 0:
            # print("Current function interpreted as 'charge' for lookup fns.")
            return 'chg'
        elif np.mean(current_arr) > 0:
            # print("Current function interpreted as 'discharge' for lookup fns.")
            return 'dchg'

    def __determine_lookup_fn(self, current_arr):
        if self.__determine_hysteresis(current_arr) == 'chg': return self.SOCfromOCV_fn_chg
        elif self.__determine_hysteresis(current_arr) == 'dchg': return self.SOCfromOCV_fn_dchg

    def KF_true(
            self,
            ocv_data=None,
            current_data=None,
            SOC_initial=None, 
            P_initial=None, 
            Q=None, 
            R=None
        ):
        """
        Estimate the State of Charge (SOC) using a Kalman filter.

        Parameters:
        - ocv_measurements: Array of ocv measurements.
        - current_measurements: Array of current measurements.
        - dt: Time step for state transition (seconds).
        - SOC_initial: Initial estimate of SOC.
        - P_initial: Initial error covariance matrix (optional).
        - Q: Process noise covariance matrix (optional).
        - R: Measurement noise covariance matrix (optional).

        Returns:
        - SOC_estimates: Estimated SOC values over time.
        """
        
        
        # Initialize parameters
        if SOC_initial is None:
            SOC_initial = self.Dataset.initial_soc_percent/100
        if ocv_data is None:
            ocv_data = self.Dataset.data.ocv["data"]
        if current_data is None:
            current_data = self.Dataset.data.current["data"]
    
        if P_initial is None:
            P_initial = np.array([[1]])  # Default initial error covariance
        if Q is None:
            Q = np.array([[0.01]])  # Default process noise covariance
        if R is None:
            R = np.array([[0.1]])         # Default measurement noise covariance

        # State transition matrix A and measurement matrix H
        A = np.array([[1]])  # State transition matrix
        B = self.Dataset.evaluation_time_step / (self.Dataset.capacity_Ah * 3600) if self.Dataset.capacity_Ah else np.array([[0]])  # Control input matrix (Δt / C_bat)
        C = np.array([[1]])  # Measurement matrix
        
        # Initialize state and covariance matrices
        x_prev = np.array([[SOC_initial]])
        P_prev = P_initial
        
        SOC_estimates = []
        other_var_tracking = []

        ocvtosoc_lookup_fn = self.__determine_lookup_fn(current_data)

        for k in range(len(ocv_data)):
            
            u = current_data[k]  # A
            
            ### Prediction Steps ###
            # 1. State Prediction Time Update
            x_pred = A @ x_prev + B * u  # Predict next state
            x_pred = np.clip(x_pred, 0, 1)
            
            # 2. Error Covariance Time Update
            P_pred = A @ P_prev @ A.T + Q  # Predict error covariance
            
            # 3. Predict System Output
            y_pred = ocvtosoc_lookup_fn(x_prev[0][0]*100)  # Predicted measurement based on predicted state
            
            ### Correction Steps ###
            
            # Measured voltage (from battery)
            V_m = ocv_data[k]  # Measured voltage in Volts
            
            # 4. Estimator Gain Matrix
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)  # Kalman gain

            # 5. State Update with Measurement
            x_new = x_pred + K @ np.array([[(V_m - y_pred)]])   # Update state estimate with measurement
            
            # 6. Error Variance Measurement Update
            P_new = (np.eye(len(K)) - K @ C) @ P_pred  # Update error covariance
            
            SOC_estimates.append(x_new[0][0])  # Store estimated SOC
            other_var_tracking.append({
                "Index" : k,
                "SOC_pred" : x_pred,
                "P_pred" : P_pred,
                "V_pred" : y_pred,
                "Gain" : K,
                "SOC_new" : x_new,
                "P_new" : P_new,
                "V_meas" : V_m,
            })
            
            # Prepare for next iteration
            x_prev, P_prev = np.clip(x_new, 0, 1), P_new

        def merge_dicts(dicts):
            merged = {}
            for d in dicts:
                for key, value in d.items():
                    merged.setdefault(key, []).append(value)
            return merged
        
        other_var_tracking_out = merge_dicts(other_var_tracking)
            
        return {
            "label": "'True' SOC [KF + modeled OCV] (%)",
            "data": SOC_estimates,
            "other": other_var_tracking_out
        }

    def linKF_CCVSOC(self, SOC_initial=None, P_initial=None, Q=None, R=None):
        """
        Estimate the State of Charge (SOC) using a Kalman filter.

        Parameters:
        - ocv_measurements: Array of ocv measurements.
        - current_measurements: Array of current measurements.
        - dt: Time step for state transition (seconds).
        - SOC_initial: Initial estimate of SOC.
        - P_initial: Initial error covariance matrix (optional).
        - Q: Process noise covariance matrix (optional).
        - R: Measurement noise covariance matrix (optional).

        Returns:
        - SOC_estimates: Estimated SOC values over time.
        """
        
        
        # Initialize parameters
        if SOC_initial is None:
            SOC_initial = self.Dataset.initial_soc_percent/100
        ccv_data = self.Dataset.data.voltage["data"]
        current_data = self.Dataset.data.current["data"]
        time_data = self.Dataset.data.time["data"]
    
        if P_initial is None:
            P_initial = np.array([[1]])  # Default initial error covariance
        if Q is None:
            Q = np.array([[0.01]])  # Default process noise covariance
        if R is None:
            R = np.array([[0.1]])         # Default measurement noise covariance

        # State transition matrix A and measurement matrix H
        A = np.array([[1]])  # State transition matrix
        B = self.Dataset.evaluation_time_step / (self.Dataset.capacity_Ah * 3600) if self.Dataset.capacity_Ah else np.array([[0]])  # Control input matrix (Δt / C_bat)
        C = np.array([[1]])  # Measurement matrix
        
        # Initialize state and covariance matrices
        x_prev = np.array([[SOC_initial]])
        P_prev = P_initial
        
        SOC_estimates = []
        other_var_tracking = []

        for k in range(len(time_data)):
            
            u = current_data[k]  # A
            V_m = ccv_data[k]  # Measured voltage (from battery)
            
            ### Prediction Steps ###
            # 1. State Prediction Time Update
            x_pred = A @ x_prev + B * u  # Predict next state
            x_pred = np.clip(x_pred, 0, 1)
            
            # 2. Error Covariance Time Update
            P_pred = A @ P_prev @ A.T + Q  # Predict error covariance
            
            # 3. Predict System Output
            if self.__determine_hysteresis(current_data) == "chg":
                y_pred = 3.7 + 0.2*x_pred
            elif self.__determine_hysteresis(current_data) == "dchg":
                y_pred = 3.7 - 0.2*x_pred
            
            ### Correction Steps ###
            
            # 4. Estimator Gain Matrix
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)  # Kalman gain

            # 5. State Update with Measurement
            x_new = x_pred + K @ (V_m - y_pred)   # Update state estimate with measurement
            
            # 6. Error Variance Measurement Update
            P_new = (np.eye(len(K)) - K @ C) @ P_pred  # Update error covariance
            
            SOC_estimates.append(x_new[0][0])  # Store estimated SOC
            other_var_tracking.append({
                "Index" : k,
                "SOC_pred" : x_pred,
                "P_pred" : P_pred,
                "V_pred" : y_pred,
                "Gain" : K,
                "SOC_new" : x_new,
                "P_new" : P_new,
                "V_meas" : V_m,
            })
            
            # Prepare for next iteration
            x_prev, P_prev = np.clip(x_new, 0, 1), P_new

        def merge_dicts(dicts):
            merged = {}
            for d in dicts:
                for key, value in d.items():
                    merged.setdefault(key, []).append(value)
            return merged
        
        other_var_tracking_out = merge_dicts(other_var_tracking)
            
        return {
            "label": "Linear KF-estimated SOC (%)",
            "data": SOC_estimates,
            "other": other_var_tracking_out
        }

    def extKF_CCVSOC(self, SOC_initial=None, P_initial=None, Q=None, R=None):
        """
        Estimate the State of Charge (SOC) using a Kalman filter.

        Parameters:
        - ocv_measurements: Array of ocv measurements.
        - current_measurements: Array of current measurements.
        - dt: Time step for state transition (seconds).
        - SOC_initial: Initial estimate of SOC.
        - P_initial: Initial error covariance matrix (optional).
        - Q: Process noise covariance matrix (optional).
        - R: Measurement noise covariance matrix (optional).

        Returns:
        - SOC_estimates: Estimated SOC values over time.
        """
        
        # Initialize parameters
        if SOC_initial is None:
             SOC_initial = self.Dataset.initial_soc_percent/100
        if P_initial is None:
            P_initial = np.array([[1]])  # Default initial error covariance
        if Q is None:
            Q = np.array([[0.01]])  # Default process noise covariance
        if R is None:
            R = np.array([[0.1]])         # Default measurement noise covariance

        # State transition matrix A and measurement matrix H
        A = np.array([[1]])  # State transition matrix
        B = self.Dataset.evaluation_time_step / (self.Dataset.capacity_Ah * 3600) if self.Dataset.capacity_Ah else np.array([[0]])  # Control input matrix (Δt / C_bat)
        C = np.array([[1]])  # Measurement/observation matrix (we'll update this dynamically)
        R_ohm = 0.06

        # Initialize state and covariance matrices
        x_prev = np.array([[SOC_initial]])
        P_prev = P_initial
        
        SOC_estimates = []
        other_var_tracking = []

        time_data = self.Dataset.data.time["data"]
        ccv_data = self.Dataset.data.voltage["data"]
        current_data = self.Dataset.data.current["data"]
        ocvtosoc_lookup_fn = self.__determine_lookup_fn(current_data)
        
        for k in range(len(time_data)):
            
            i_in = current_data[k]  # A
            z_Vmeas = ccv_data[k]  # Measured voltage (from battery)
            
            #======================================
            # PREDICTION STEPS
            #======================================
            
            # ----------- 1. State Prediction Time Update
            # SOC state predicted based on prior estimate + what's been input since then (current*control input matrix B)

            x_pred = A @ x_prev + B * i_in  # Predict next state
            x_pred = np.clip(x_pred, 0, 1)
            
            # ----------- 2. Error Covariance Time Update
            P_pred = A @ P_prev @ A.T + Q  # Predict error covariance
            
            # ----------- 3. Predict System Output
            # In simplest form, you would have something like y = V_m - C@x_pred, which we use as our INNOVATION (residual)
            # Obviously, with V_m being CCV and x_pred being SOC, residual will not be very representative and reducing may not yield accurate results
            # To be more accurate, we demonstrate a more detailed mapping of SOC to a predicted voltage

            z_pOCV = ocvtosoc_lookup_fn(x_prev[0][0]*100)  # predicted OCV <--from-- predicted SOC using lookup table

            # We also know that returned result is OCV, but measured voltage is a CCV, so we must also model this to get a predicted CCV

            z_pCCV = z_pOCV - (i_in * R_ohm)   # V_CCV = V_OCV - IR
            
            # And now we can find the residual
            y = z_Vmeas - z_pCCV

            # Update observation matrix C (derivative of OCV wrt SOC)
            # We approximate the derivative using finite differences
            delta_soc = 0.001  # Small change in SOC for numerical derivative
            C = np.array([[(ocvtosoc_lookup_fn((x_pred[0][0] + delta_soc)*100) - ocvtosoc_lookup_fn(x_pred[0][0]*100)) / delta_soc]])


            #======================================
            # CORRECTION STEPS
            #======================================
            
            # ----------- 4. Estimator Gain Matrix (Kalman Gain)
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)

            # ----------- 5. State Update with Measurement
            x_new = x_pred + K @ np.array([y])
            
            # ----------- 6. Error Variance Measurement Update
            P_new = (np.eye(len(K)) - K @ C) @ P_pred
            
            SOC_estimates.append(x_new[0][0])  # Store estimated SOC
            other_var_tracking.append({
                "k" : k,
                "x_pred (SOC prediction)" : x_pred,
                "P_pred" : P_pred,
                "y (innovation)" : y,
                "K" : K,
                "x_new (SOC estimation)" : x_new,
                "P_new" : P_new,
                "z (V_meas)" : z_Vmeas,
                "z (V_OCV)" : z_pOCV,
                "z (V_CCV)" : z_pCCV,
            })
            
            # Prepare for next iteration
            x_prev, P_prev = np.clip(x_new, 0, 1), P_new

        def merge_dicts(dicts):
            merged = {}
            for d in dicts:
                for key, value in d.items():
                    merged.setdefault(key, []).append(value)
            return merged
        
        other_var_tracking_out = merge_dicts(other_var_tracking)
            
        return {
            "label": "Extended KF-estimated SOC (%)",
            "data": SOC_estimates,
            "other": other_var_tracking_out
        }

    # def extKF_VSOC(self, SOC_initial=None, P_initial=None, Q=None, R=None):
    #     """
    #     Estimate the State of Charge (SOC) using an Extended Kalman Filter (EKF).

    #     Parameters:
    #     - ocv_measurements: Array of OCV measurements.
    #     - current_measurements: Array of current measurements.
    #     - dt: Time step for state transition (seconds).
    #     - SOC_initial: Initial estimate of SOC.
    #     - P_initial: Initial error covariance matrix (optional).
    #     - Q: Process noise covariance matrix (optional).
    #     - R: Measurement noise covariance matrix (optional).

    #     Returns:
    #     - SOC_estimates: Estimated SOC values over time.
    #     """
        
    #     # Initialize parameters
    #     if SOC_initial is None:
    #          SOC_initial = self.Dataset.initial_soc_percent/100
    #     if P_initial is None:
    #         P_initial = np.array([[1]])  # Default initial error covariance
    #     if Q is None:
    #         Q = np.array([[0.01]])  # Default process noise covariance
    #     if R is None:
    #         R = np.array([[0.1]])         # Default measurement noise covariance

    #     # Initialize state and covariance matrices
    #     x_prev = np.array([[SOC_initial]])
    #     P_prev = P_initial
        
    #     SOC_estimates = []
    #     other_var_tracking = []
        
    #     ocv_data = self.Dataset.data.ocv["data"]
    #     current_data = self.Dataset.data.current["data"]
    #     ocvtosoc_lookup_fn = self.__determine_lookup_fn(current_data)
        
    #     for k in range(len(ocv_data)):
            
    #         u = current_data[k]  # Current input
            
    #         ### Prediction Steps ###
    #         # 1. State Prediction Time Update
    #         x_pred = x_prev + (self.Dataset.evaluation_time_step / (self.Dataset.capacity_Ah * 3600)) * u  # Non-linear state transition
            
    #         # 2. Error Covariance Time Update
    #         P_pred = P_prev + Q  # Predict error covariance
            
    #         ### Correction Steps ###
            
    #         # Measured voltage (from battery)
    #         V_m = ocv_data[k]  # Measured voltage in Volts
            
    #         # 3. Predict System Output
    #         y_pred = ocvtosoc_lookup_fn(x_pred[0][0] * 100)  # Predicted measurement based on predicted state
            
    #         # Calculate Jacobians
    #         H_jacobian = np.array(([[ocvtosoc_lookup_fn(x_pred[0][0] * 100 + 1e-6) - ocvtosoc_lookup_fn(x_pred[0][0] * 100)]]) / 1e-6)  # Jacobian of measurement model
            
    #         # 4. Estimator Gain Matrix
    #         K = P_pred @ H_jacobian.T @ np.linalg.inv(H_jacobian @ P_pred @ H_jacobian.T + R)  # Kalman gain

    #         # 5. State Update with Measurement
    #         x_new = x_pred + K @ np.array([[(V_m - y_pred)]])   # Update state estimate with measurement
            
    #         # 6. Error Variance Measurement Update
    #         P_new = (np.eye(len(K)) - K @ H_jacobian) @ P_pred  # Update error covariance
            
    #         SOC_estimates.append(np.clip(x_new[0][0], 0, 1))  # Store estimated SOC
    #         other_var_tracking.append({
    #             "k": k,
    #             "x_pred": x_pred,
    #             "P_pred": P_pred,
    #             "y_pred": y_pred,
    #             "K": K,
    #             "x_new": x_new,
    #             "P_new": P_new,
    #             "z": V_m,
    #         })
            
    #         # Prepare for next iteration
    #         x_prev, P_prev = x_new, P_new

    #     def merge_dicts(dicts):
    #         merged = {}
    #         for d in dicts:
    #             for key, value in d.items():
    #                 merged.setdefault(key, []).append(value)
    #         return merged
        
    #     other_var_tracking_out = merge_dicts(other_var_tracking)
            
    #     return {
    #         "label": "State of Charge via Extended Kalman Filter (%)",
    #         "data": SOC_estimates,
    #         "other": other_var_tracking_out
    #     }


# def simple_KF_soc(voltage_measurements, current_measurements, dt, 
#                       SOC_initial=0.5, P_initial=None, Q=None, R=None):
#     """
#     Estimate the State of Charge (SOC) using a Kalman filter.

#     Parameters:
#     - voltage_measurements: Array of voltage measurements.
#     - current_measurements: Array of current measurements.
#     - dt: Time step for state transition.
#     - SOC_initial: Initial estimate of SOC.
#     - P_initial: Initial error covariance matrix (optional).
#     - Q: Process noise covariance matrix (optional).
#     - R: Measurement noise covariance matrix (optional).

#     Returns:
#     - SOC_estimates: Estimated SOC values over time.
#     """
    
#     # Initialize parameters
#     if P_initial is None:
#         P_initial = np.eye(2)  # Default initial error covariance
#     if Q is None:
#         Q = np.diag([0.01, 0.01])  # Default process noise covariance
#     if R is None:
#         R = np.diag([0.1])          # Default measurement noise covariance

#     # State transition matrix A and measurement matrix H
#     A = np.array([[1, dt], [0, 1]])  # State transition model
#     H = np.array([[1, 0]])            # Measurement model
    
#     # Initialize state and covariance matrices
#     x_prev = np.array([SOC_initial, voltage_measurements[0]])  # [SOC, Voltage, other states, ...]
#     P_prev = P_initial
    
#     SOC_estimates = []
#     other_var_tracking = []

#     def merge_dicts(dicts):
#         merged = {}
#         for d in dicts:
#             for key, value in d.items():
#                 merged.setdefault(key, []).append(value)
#         return merged
    
#     for k in range(len(voltage_measurements)):
#         ### Prediction Steps ###
        
#         # 1. State Prediction Time Update
#         x_pred = A @ x_prev + np.array([0, current_measurements[k]])  # Predict next state
        
#         # 2. Error Covariance Time Update
#         P_pred = A @ P_prev @ A.T + Q  # Predict error covariance
        
#         # 3. Predict System Output
#         y_pred = H @ x_pred  # Predicted measurement based on predicted state
        
#         ### Correction Steps ###
#         # 4. Estimator Gain Matrix
#         y_tilde = voltage_measurements[k] - y_pred  # Measurement residual
#         S = H @ P_pred @ H.T + R  # Residual covariance
#         K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        
#         # 5. State Update with Measurement
#         x_new = x_pred + K @ y_tilde  # Update state estimate with measurement
        
#         # 6. Error Variance Measurement Update
#         P_new = (np.eye(len(K)) - K @ H) @ P_pred  # Update error covariance

#         SOC_estimates.append(x_new[0])  # Store estimated SOC
#         other_var_tracking.append({
#             "k" : k,
#             "x_pred" : x_pred,
#             "P_pred" : P_pred,
#             "y_pred" : y_pred,
#             "y_tilde" : y_tilde,
#             "S" : S,
#             "K" : K,
#             "x_new" : x_new,
#             "P_new" : P_new,
#         })
        
#         # Prepare for next iteration
#         x_prev, P_prev = x_new, P_new

#     other_var_tracking_out = merge_dicts(other_var_tracking)
        
#     return {
#         "label": "State of Charge via simple KF (%)",
#         "data": SOC_estimates,
#         "other": other_var_tracking_out
#     }
