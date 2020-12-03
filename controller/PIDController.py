

from controller.Controller import Controller
import numpy as np

class PIDController(Controller):
    def __init__(self, dof, params):
        if len(params) > 2:
            self.kd_ = params[2]
        else:
            self.kd_ = 0
        if len(params) > 1:
            self.ki_ = params[1]
        else:
            self.ki_ = 0
        if len(params) > 0:
            self.kp_ = params[0]
        else:
            raise AttributeError

        self.int_error = np.zeros((dof,1))
        self.prev_error = np.zeros((dof,1))

    def calc_action_forces(self, ref, e, de, dt):
        self.int_error += e * dt

        #action_forces = ref + self.kp_*e + self.ki_*self.int_error + self.kd_*de
        action_forces = self.kp_*e + self.ki_*self.int_error + self.kd_*de
        return action_forces




