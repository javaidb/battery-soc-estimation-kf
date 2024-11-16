import numpy as np

def coulomb_counting(current, time, initial_soc_percent, capacity):
    initial_soc = initial_soc_percent/100
    dt = np.diff(time)
    accumulated_charge = np.cumsum(current[1:] * dt)
    soc = initial_soc - accumulated_charge / (capacity * 3600)
    soc_with_initial = np.insert(soc, 0, initial_soc)
    return {
        "label": "State of Charge via Coul. Cnt (%)",
        "data": np.clip(soc_with_initial, 0, 1)
    }

def linKF_VSOC(ocv_measurements, current_measurements, dt, OCV_lookup_fn,
                      capacity_Ah=5, SOC_initial=0.5, P_initial=None, Q=None, R=None):
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
    if P_initial is None:
        P_initial = np.array([[1]])  # Default initial error covariance
    if Q is None:
        Q = np.array([[0.01]])  # Default process noise covariance
    if R is None:
        R = np.array([[0.1]])         # Default measurement noise covariance

    # State transition matrix A and measurement matrix H
    A = np.array([[1]])  # State transition matrix
    B = dt / (capacity_Ah * 3600) if capacity_Ah else np.array([[0]])  # Control input matrix (Δt / C_bat)
    C = np.array([[1]])  # Measurement matrix
    
    # Initialize state and covariance matrices
    x_prev = np.array([[SOC_initial]])
    P_prev = P_initial
    
    SOC_estimates = []
    other_var_tracking = []

    def merge_dicts(dicts):
        merged = {}
        for d in dicts:
            for key, value in d.items():
                merged.setdefault(key, []).append(value)
        return merged
    
    for k in range(len(ocv_measurements)):
        
        # Control input (current applied to the battery)
        u = current_measurements[k]  # Current in Amperes
        # Measured voltage (from battery)
        V_m = ocv_measurements[k]  # Measured voltage in Volts
        # Open Circuit Voltage (OCV) reference for SOC estimation from voltage
        z = OCV_lookup_fn(V_m)/100  # Measured SOC derived from voltage (assumed)
        
        ### Prediction Steps ###
        # 1. State Prediction Time Update
        x_pred = A @ x_prev + B * u  # Predict next state
        
        # 2. Error Covariance Time Update
        P_pred = A @ P_prev @ A.T + Q  # Predict error covariance
        
        # 3. Predict System Output
        y_pred = C @ x_pred  # Predicted measurement based on predicted state
        
        ### Correction Steps ###
        # 4. Estimator Gain Matrix
        K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)  # Kalman gain
        
        # 5. State Update with Measurement
        x_new = x_pred + K @ (z - y_pred)   # Update state estimate with measurement
        
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
            "z" : z,
        })
        
        # Prepare for next iteration
        x_prev, P_prev = np.clip(x_new, 0, 1), P_new
    other_var_tracking_out = merge_dicts(other_var_tracking)
        
    return {
        "label": "State of Charge via simple KF (%)",
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
