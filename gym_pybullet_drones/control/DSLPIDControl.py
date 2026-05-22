import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class tuned for SimAstra 8kg Tactical Drone."""

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P and self.DRONE_MODEL != DroneModel.HEAVY:
            print("[ERROR] DSLPIDControl requires DroneModel.CF2X, CF2P, or HEAVY")
            exit()

        # =============================================================
        # TUNING FOR YOUR URDF (8KG, KF=7.84e-7, Izz=1.0)
        # =============================================================
        if self.DRONE_MODEL == DroneModel.HEAVY:
            
            # --- POSITION CONTROL (Force in Newtons) ---
            # 8kg mass requires significant P gain to correct position errors.
            self.P_COEFF_FOR = np.array([18.0, 18.0, 35.0]) 
            self.I_COEFF_FOR = np.array([0.5, 0.5, 2.0])
            self.D_COEFF_FOR = np.array([12.0, 12.0, 15.0]) 

            # --- ATTITUDE CONTROL (Torque) ---
            # Your Inertia (Ixx=0.5, Izz=1.0) is very high. 
            # We need MASSIVE torque gains to rotate this heavy cylinder.
            self.P_COEFF_TOR = np.array([150000., 150000., 200000.])
            self.I_COEFF_TOR = np.array([500., 500., 500.])
            self.D_COEFF_TOR = np.array([80000., 80000., 80000.])

            # --- MOTOR LIMITS ---
            # Your T2W is 2.25. 
            # Max Thrust = 2.25 * 78.4N = ~176N
            # Max RPM = sqrt( (176/4) / 7.84e-7 ) = ~7500 RPM
            # We set limit to 9000 to allow headroom for maneuvering.
            self.PWM2RPM_SCALE = 1.0 
            self.PWM2RPM_CONST = 0.0
            self.MIN_PWM = 0
            self.MAX_PWM = 9000 
            self.MASS = 15.0

        # =============================================================
        # STANDARD CRAZYFLIE TUNING
        # =============================================================
        else:
            self.P_COEFF_FOR = np.array([.4, .4, 1.25])
            self.I_COEFF_FOR = np.array([.05, .05, .05])
            self.D_COEFF_FOR = np.array([.2, .2, .5])
            self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
            self.I_COEFF_TOR = np.array([.0, .0, 500.])
            self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
            self.PWM2RPM_SCALE = 0.2685
            self.PWM2RPM_CONST = 4070.3
            self.MIN_PWM = 20000
            self.MAX_PWM = 65535
            self.MASS = 0.027

        # MIXER CONFIGURATION
        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.HEAVY:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    def reset(self):
        super().reset()
        self.last_rpy = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel)
        rpm = self._dslPIDAttitudeControl(control_timestep, thrust, cur_quat, computed_target_rpy, target_rpy_rates)
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        # FIX: Multiply GRAVITY by MASS (9.8 * 15.0 = 147N) to get Force, not Acceleration
        gravity_force = np.array([0, 0, self.GRAVITY * self.MASS])
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) + np.multiply(self.D_COEFF_FOR, vel_e) + gravity_force
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        # Robustness: Check for zero thrust
        norm_thrust = np.linalg.norm(target_thrust)
        if norm_thrust < 1e-6:
            target_z_ax = np.array([0,0,1])
        else:
            target_z_ax = target_thrust / norm_thrust
            
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        
        # Robustness: Check for collinear vectors
        cross_prod = np.cross(target_z_ax, target_x_c)
        norm_cross = np.linalg.norm(cross_prod)
        if norm_cross < 1e-6:
            target_y_ax = np.array([0,1,0])
        else:
            target_y_ax = cross_prod / norm_cross
            
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        return thrust, target_euler, pos_e

    def _dslPIDAttitudeControl(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        # Direct matrix conversion to avoid quaternion issues
        target_rotation = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) + np.multiply(self.D_COEFF_TOR, rpy_rates_e) + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    def _one23DInterface(self, thrust):
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
