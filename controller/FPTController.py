

from controller.Controller import Controller
import numpy as np
import math
from plant.PumaArm3DOF import PumaArm3DOF

class FPTController(Controller):
    def __init__(self, dim, params, dyn_model_params):
        self.l = params[0]
        self.D = params[1]
        self.A = params[2]

        self.dim = dim
        self.ddq_def_prev = np.zeros((dim, 1))
        self.ddq_real_prev = np.zeros((dim, 1))
        self.int_error = np.zeros((dim, 1))

        self.dyn_model = PumaArm3DOF(dyn_model_params)

        self.x_star = 0
        self.init_x_star()

    def calc_action_forces(self, ref, e, de, dt):
        self.int_error += e*dt
        des_ddq = self.kinBlock(ref, self.int_error, e, de)
        def_ddq = self.G(des_ddq)
        action_forces = self.dyn_model.calc_required_focres(def_ddq)
        self.ddq_def_prev = def_ddq
        return [des_ddq, def_ddq, action_forces]

    def Fdef(self, x):
        return (x / 2) + self.D

    def init_x_star(self):
        for i in range(100):
            self.x_star = self.Fdef(self.x_star)

    def G(self, ddq_des):
        h = self.ddq_real_prev-ddq_des
        h_norm = np.linalg.norm(h)
        if h_norm > 1e-10:
            return (self.Fdef(self.A*h_norm+self.x_star)-self.x_star)*(h/h_norm)+self.ddq_def_prev
        else:
            return self.ddq_def_prev

    # A kinematic block of function
    def kinBlock(self, ddq_ref, eint, e, de):
        return  ddq_ref+(math.pow(self.l,3)*eint+3*math.pow(self.l,2)*e+3*self.l*de)

    def update_real_effect(self, real_effect):
        self.ddq_real_prev = real_effect