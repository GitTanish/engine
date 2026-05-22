import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class HeavyDSLPIDControl(DSLPIDControl):
    """
    Heavy Drone Controller (15kg)
    """
    def __init__(self, drone_model: DroneModel=DroneModel.HEAVY, g: float=9.8):
        super().__init__(drone_model=drone_model, g=g)
        self.MASS = 15.0
        
        # PID Gains (Step 1 Tuning)
        self.P_COEFF_FOR = np.array([6.0, 6.0, 18.75])
        self.I_COEFF_FOR = np.array([0.2, 0.2, 0.2])
        self.D_COEFF_FOR = np.array([12.0, 12.0, 8.0])
        
        self.P_COEFF_TOR = np.array([60000., 60000., 60000.])
        self.I_COEFF_TOR = np.array([200., 200., 200.])
        self.D_COEFF_TOR = np.array([20000., 20000., 20000.])
        
        # Motor Constants
        self.KF = 7.84e-7
        self.KM = 3.92e-8
        
        self.reset()
