import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class HeavyDSLPIDControl(DSLPIDControl):
    """
    Heavy Drone Controller (15kg) - STABILIZED (Reverted Tuning)
    """
    def __init__(self, drone_model: DroneModel=DroneModel.HEAVY, g: float=9.8):
        super().__init__(drone_model=drone_model, g=g)
        self.MASS = 15.0
        
        # --- POSITION CONTROL (Safe Tuning) ---
        # P=20: Sufficient to correct drift, relies on Gravity Comp for lift.
        # I=0.5: Standard integral to fix small steady-state errors.
        # D=12: Dampens oscillation without fighting the lift.
        self.P_COEFF_FOR = np.array([18.0, 18.0, 20.0]) 
        self.I_COEFF_FOR = np.array([0.5, 0.5, 0.5])
        self.D_COEFF_FOR = np.array([12.0, 12.0, 12.0]) 
        
        # --- ATTITUDE CONTROL ---
        # High torque for 15kg rotational inertia
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([0., 0., 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        
        # Motor Constants
        self.KF = 7.84e-7
        self.KM = 3.92e-8
        
        self.reset()
