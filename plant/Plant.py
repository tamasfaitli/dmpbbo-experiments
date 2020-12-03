##

import numpy as np

class Plant:
    def __init__(self, dof):
        assert(isinstance(dof,int))
        self.dof_   = dof
        self.q_     = np.zeros((dof,1))
        self.dq_    = np.zeros((dof,1))
        self.ddq_   = np.zeros((dof,1))


    def init_state(self, q, dq, ddq):
        self.q_   = q
        self.dq_  = dq
        self.ddq_ = ddq

    def update_state(self, ddq, dt):
        q = np.array(self.q_)
        dq = np.array(self.dq_)

        self.ddq_ = np.array(ddq)
        self.dq_  = np.array(dq + (ddq * dt))
        self.q_   = np.array(q + (dq*dt))


    def get_current_state(self):
        return [self.q_, self.dq_, self.ddq_]

    def init_parameters(self, params):
        raise NotImplementedError('subclasses must override init_parameters()')

    def calc_effect_of_action_forces(self, action_forces):
        raise NotImplementedError

    def calc_required_focres(self, desired_effect):
        raise NotImplementedError