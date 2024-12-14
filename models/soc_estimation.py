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
            "label": "State of Charge via Coul. Cnt (%)",
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

    def linKF_OCVSOC(
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
            "label": "State of Charge via simple KF (%)",
            "data": SOC_estimates,
            "other": other_var_tracking_out
        }

    def linKF_OCVSOC_v2(self, SOC_initial=None, P_initial=None, Q=None, R=None):
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

        R_0 = 0.01
        # State transition matrix A and measurement matrix H
        A = np.array([[1]])  # State transition matrix
        B = self.Dataset.evaluation_time_step / (self.Dataset.capacity_Ah * 3600) if self.Dataset.capacity_Ah else np.array([[0]])  # Control input matrix (Δt / C_bat)
        C = np.array([[0.7]])  # Measurement matrix
        D = -R_0 
        
        # Initialize state and covariance matrices
        x_prev = np.array([[SOC_initial]])
        P_prev = P_initial
        
        SOC_estimates = []
        other_var_tracking = []
        
        ocv_data = self.Dataset.data.ocv["data"]
        current_data = self.Dataset.data.current["data"]
        
        for k in range(len(ocv_data)):
            
            u = current_data[k]  # A
            
            ### Prediction Steps ###
            # 1. State Prediction Time Update
            x_pred = A @ x_prev + B * u  # Predict next state
            x_pred = np.clip(x_pred, 0, 1)
            
            # 2. Error Covariance Time Update
            P_pred = A @ P_prev @ A.T + Q  # Predict error covariance
            
            # 3. Predict System Output
            y_pred = C*x_pred + D*u  # Predicted measurement based on predicted state
            
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
                "k" : k,
                "x_pred" : x_pred,
                "P_pred" : P_pred,
                "y_pred" : y_pred,
                "K" : K,
                "x_new" : x_new,
                "P_new" : P_new,
                "z" : V_m,
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
            "label": "State of Charge via simple KF (%)",
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
        
        ccv_data = self.Dataset.data.voltage["data"]
        current_data = self.Dataset.data.current["data"]
        ocvtosoc_lookup_fn = self.__determine_lookup_fn(current_data)
        
        for k in range(len(ccv_data)):
            
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
            V_m = ccv_data[k]  # Measured voltage in Volts
            
            # 4. Estimator Gain Matrix
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)  # Kalman gain

            # 5. State Update with Measurement
            x_new = x_pred + K @ np.array([[(V_m - y_pred)]])   # Update state estimate with measurement
            
            # 6. Error Variance Measurement Update
            P_new = (np.eye(len(K)) - K @ C) @ P_pred  # Update error covariance
            
            SOC_estimates.append(x_new[0][0])  # Store estimated SOC
            other_var_tracking.append({
                "k" : k,
                "x_pred" : x_pred,
                "P_pred" : P_pred,
                "y_pred" : y_pred,
                "K" : K,
                "x_new" : x_new,
                "P_new" : P_new,
                "z" : V_m,
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
            "label": "State of Charge via simple KF (%)",
            "data": SOC_estimates,
            "other": other_var_tracking_out
        }

    def extKF_VSOC(self, SOC_initial=None, P_initial=None, Q=None, R=None):
        """
        Estimate the State of Charge (SOC) using an Extended Kalman Filter (EKF).

        Parameters:
        - ocv_measurements: Array of OCV measurements.
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

        # Initialize state and covariance matrices
        x_prev = np.array([[SOC_initial]])
        P_prev = P_initial
        
        SOC_estimates = []
        other_var_tracking = []
        
        ocv_data = self.Dataset.data.ocv["data"]
        current_data = self.Dataset.data.current["data"]
        ocvtosoc_lookup_fn = self.__determine_lookup_fn(current_data)
        
        for k in range(len(ocv_data)):
            
            u = current_data[k]  # Current input
            
            ### Prediction Steps ###
            # 1. State Prediction Time Update
            x_pred = x_prev + (self.Dataset.evaluation_time_step / (self.Dataset.capacity_Ah * 3600)) * u  # Non-linear state transition
            
            # 2. Error Covariance Time Update
            P_pred = P_prev + Q  # Predict error covariance
            
            ### Correction Steps ###
            
            # Measured voltage (from battery)
            V_m = ocv_data[k]  # Measured voltage in Volts
            
            # 3. Predict System Output
            y_pred = ocvtosoc_lookup_fn(x_pred[0][0] * 100)  # Predicted measurement based on predicted state
            
            # Calculate Jacobians
            H_jacobian = np.array(([[ocvtosoc_lookup_fn(x_pred[0][0] * 100 + 1e-6) - ocvtosoc_lookup_fn(x_pred[0][0] * 100)]]) / 1e-6)  # Jacobian of measurement model
            
            # 4. Estimator Gain Matrix
            K = P_pred @ H_jacobian.T @ np.linalg.inv(H_jacobian @ P_pred @ H_jacobian.T + R)  # Kalman gain

            # 5. State Update with Measurement
            x_new = x_pred + K @ np.array([[(V_m - y_pred)]])   # Update state estimate with measurement
            
            # 6. Error Variance Measurement Update
            P_new = (np.eye(len(K)) - K @ H_jacobian) @ P_pred  # Update error covariance
            
            SOC_estimates.append(np.clip(x_new[0][0], 0, 1))  # Store estimated SOC
            other_var_tracking.append({
                "k": k,
                "x_pred": x_pred,
                "P_pred": P_pred,
                "y_pred": y_pred,
                "K": K,
                "x_new": x_new,
                "P_new": P_new,
                "z": V_m,
            })
            
            # Prepare for next iteration
            x_prev, P_prev = x_new, P_new

        def merge_dicts(dicts):
            merged = {}
            for d in dicts:
                for key, value in d.items():
                    merged.setdefault(key, []).append(value)
            return merged
        
        other_var_tracking_out = merge_dicts(other_var_tracking)
            
        return {
            "label": "State of Charge via Extended Kalman Filter (%)",
            "data": SOC_estimates,
            "other": other_var_tracking_out
        }


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
