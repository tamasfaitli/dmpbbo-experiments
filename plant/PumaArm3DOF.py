

from plant.Plant import Plant
import numpy as np
import math

DEG = math.pi/180

class PumaArm3DOF(Plant):
    def __init__(self, params):
        assert(len(params)==7)
        super().__init__(3) # degree of freedom is given and its 3
        self.init_parameters(params)

    def init_parameters(self, params):
        self.g_  = params[0] #gravity term
        self.th_ = params[1]
        self.m2_ = params[2]
        self.m3_ = params[3]
        self.l1_ = params[4]
        self.l2_ = params[5]
        self.l3_ = params[6]

    def calc_effect_of_action_forces(self, action_forces):
        [H, h] = self.__calc_model_matrices()
        return np.matmul(np.linalg.inv(H),(action_forces-h))

    def calc_required_focres(self, desired_effect):
        [H, h] = self.__calc_model_matrices()
        return np.matmul(H,desired_effect)+h

    def __calc_model_matrices(self):
        q2 = self.q_[1,0]
        q3 = self.q_[2,0]
        dq1 = self.dq_[0, 0]
        dq2 = self.dq_[1, 0]
        dq3 = self.dq_[2, 0]
        s2 = math.sin(q2)
        s3 = math.sin(q3)
        s23 = math.sin(q2 + q3)
        c2 = math.cos(q2)
        c3 = math.cos(q3)
        c23 = math.cos(q2 + q3)
        H = np.zeros((3, 3))
        H[0, 0] = self.th_ + 1 / 4 * self.m2_ * self.l2_ * self.l2_ * c2 * c2 + 1 / 4 * self.m3_ * self.l3_ * self.l3_ * c23 * c23 + self.m3_ * self.l2_ * self.l2_ * c2 * c2 + 1 / 2 * self.m3_ * self.l2_ * self.l3_ * c23 * c2
        H[1, 1] = 1 / 4 * self.m2_ * self.l2_ * self.l2_ + 1 / 4 * self.m3_ * self.l3_ * self.l3_ + self.m3_ * self.l2_ * self.l2_ + 1 / 2 * self.m3_ * self.l3_ * self.l2_ * c3
        H[1, 2] = 1 / 4 * self.m3_ * self.l3_ * self.l3_ + 1 / 4 * self.m3_ * self.l3_ * self.l2_ * c3
        H[2, 1] = 1 / 4 * self.m3_ * self.l3_ * self.l3_ + 1 / 4 * self.m3_ * self.l3_ * self.l2_ * c3
        H[2, 2] = 1 / 4 * self.m3_ * self.l3_ * self.l3_
        h = np.zeros((3,1))
        h[0, 0] = -1 / 2 * self.m2_ * self.l2_ * self.l2_ * c2 * s2 * dq1 * dq2 - 1 / 2 * self.m3_ * self.l3_ * self.l3_ * c23 * s23 * dq1 * (
                    dq2 + dq3) - 2 * self.m3_ * self.l2_ * self.l2_ * c2 * s2 * dq1 * dq2 + 1 / 2 * self.m3_ * self.l2_ * self.l3_ * (
                          -s23 * c2 * dq1 * (dq2 + dq3) - c23 * s2 * dq1 * dq2)
        h[1, 0] = -1 / 4 * self.m3_ * self.l3_ * self.l2_ * s3 * dq3 * dq3 + 1 / 4 * self.m2_ * self.l2_ * self.l2_ * c2 * s2 * dq1 * dq1 + 1 / 4 * self.m3_ * self.l3_ * self.l3_ * c23 * s23 * dq1 * dq1 + self.m3_ * self.l2_ * self.l2_ * c2 * s2 * dq1 * dq1 + 1 / 4 * self.m3_ * self.l2_ * self.l3_ * (
                    s23 * c2 + c23 * s2) * dq1 * dq1 + 1 / 2 * self.l2_ * self.m2_ * self.g_ * c2 + self.m3_ * self.g_ * (self.l2_ * c2 + 1 / 2 * self.l3_ * c23)
        h[1, 0] = h[1, 0] - 1 / 2 * self.m3_ * self.l3_ * self.l2_ * s3 * dq2 * dq2
        h[2, 0] = 1 / 4 * self.m3_ * (self.l3_ * self.l3_ * c23 * s23 + self.l3_ * self.l2_ * s23 * c2) * dq1 * dq1 + 1 / 2 * self.m3_ * self.g_ * self.l3_ * c23
        return H, h
