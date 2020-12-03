#################################################################################################################
#  Author:      Tamas Faitli                                                                                    #
#  Date:        28/04/2020                                                                                      #
#  Name:        run_dmp_experiments.py                                                                          #
#  Description: Training DMP using dmpbbo library and existing trajectory information.                          #
#################################################################################################################


#################################################################################################################
#   IMPORTS                                                                                                     #
#################################################################################################################
import os
import sys
import time
#import subprocess
import re
#import unittest
import math

#ADDITIONAL
from dmpbbo_lib.dmp.dmp_plotting import *
from dmpbbo_lib.dmp.Dmp import *
from dmpbbo_lib.dmp.Trajectory import *
from dmpbbo_lib.functionapproximators.FunctionApproximatorLWR import *
from dmpbbo_lib.functionapproximators.FunctionApproximatorRBFN import *
from plant.PumaArm3DOF import PumaArm3DOF
from controller.PIDController import PIDController
from controller.FPTController import FPTController

#################################################################################################################
#   DEFAULT CONFIG                                                                                              #
#################################################################################################################
CONF_DEBUG                              = False
CONF_SAVEFIGS                           = False
CONF_PLOT_DMP_WITH_LIBPROVIDED_FUNCTION = False
CONF_PRINT_NONFILTERED_DATA             = False
CONF_INJECT_PHASE_STOP                  = True

#################################################################################################################
#   ARGUMENTS                                                                                                   #
#################################################################################################################

#################################################################################################################
#   GLOBAL CONSTANTS                                                                                                  #
#################################################################################################################
FUNC_APPROX_TYPES = ['LWR','RBFN'] #defined in script -> DO NOT CHANGE

#################################################################################################################
#   PARAMETERS                                                                                                  #
#################################################################################################################
PAR_FILE_NAME               = 'trajectory/data/matlab_gen_for_puma3dof.txt'
PAR_SIGMOID_MR              = -1    # value for max rate of change within the sigmoid dynamical system
PAR_FUNC_APPR_TYPE          = 0     # 0 for LWR and 1 for RBFN method
PAR_NO_OF_BASIS             = 35     # number of basis functions used for the function approximators
PAR_APPLY_LP_FILTER         = False  # applying low-pass filter on demonstration data
PAR_CUTOFF_FREQ             = 0.9  # cutoff frequency used for the LP filter
PAR_FIG_NAMES               = "closed_loop_system_phase_stopping"
PAR_CONTROLLER_TYPE         = "FPT"  # either PID or FPT
PAR_ERROR_TRESHOLD          = 0.04  # 0.004

# plant parameters
PAR_G       = 9.81 # [m/s²]
PAR_THETA   = 20.0   # [kg*m²] second moment of inertia for rotating base component
PAR_M2      = 15.0  # [kg] ..
PAR_M3      = 9.0   # [kg]
PAR_L1      = 1.5 # [m]
PAR_L2      = 0.8 # [m]
PAR_L3      = 0.5 # [m]

# approximated model
PAR_aG       = 10 # [m/s²]
PAR_aTHETA   = 22.0   # [kg*m²] second moment of inertia for rotating base component
PAR_aM2      = 15.2  # [kg] ..
PAR_aM3      = 8.5   # [kg]
PAR_aL1      = 1.55 # [m]
PAR_aL2      = 0.75 # [m]
PAR_aL3      = 0.6 # [m]

# controller parameters
#PID
PAR_KP = 450
PAR_KI = 120
PAR_KD = 0.8*math.sqrt(PAR_KP)

#FPT
PAR_L = 2.6 #1.1
PAR_D = 0.3 #0.4
PAR_A = -2.0 #-1.2


#################################################################################################################
#   CLASS DEFINITIONS                                                                                           #
#################################################################################################################

#################################################################################################################
#   GLOBAL VARIABLES                                                                                            #
#################################################################################################################

#################################################################################################################
#   FUNCTION DEFINITIONS                                                                                        #
#################################################################################################################
# prep_data takes
def prep_data(raw):
    ts = []
    y = []
    yd = []
    ydd = []
    rows = raw.split('\n')

    # calc dimensions
    parse_data_from_line = "([0-9.-]+)"
    matches = re.findall(parse_data_from_line,rows[0])
    dim = int((len(matches)-1)/3)

    # parse recordings for each time stamp
    for row in rows:
        # parse numerical values from line
        row_matches = re.findall(parse_data_from_line,row)

        # skipping empty lines (e.g. last line in text file is usually empty when it's generated)
        if len(row_matches) == 0:
            continue

        # add measurements corresponding to current time stamp to the containers
        ts.append(float(row_matches[0]))
        y.append([float(x) for x in row_matches[1:1+dim]])
        yd.append([float(x) for x in row_matches[1+dim:1+(2*dim)]])
        ydd.append([float(x) for x in row_matches[1+(2*dim):1+(3*dim)]])

    # return the constructed Trajectory object using numpy matrices
    return Trajectory(np.array(ts),np.array(y),np.array(yd),np.array(ydd))

def plot_trajectories(dim,dem_data,rep_data,orig_mdata=None):
    for i in range(dim):
        leg = []
        fig = plt.figure()
        axs = [fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)]

        # demonstration
        x   = dem_data[:, 0]
        y   = dem_data[:, i+1]
        yd  = dem_data[:, dim + i+1]
        ydd = dem_data[:, (2 * dim) + i+1]

        # reproduction
        rx   = rep_data[:, 0]
        ry   = rep_data[:, i+1]
        ryd  = rep_data[:, dim + i+1]
        rydd = rep_data[:, (2 * dim) + i+1]

        # non-filtered data

        ox   = orig_mdata[:, 0]
        oy   = orig_mdata[:, i+1]
        oyd  = orig_mdata[:, dim + i+1]
        oydd = orig_mdata[:, (2 * dim) + i+1]

        axs[0].plot(ox, oy)
        axs[1].plot(ox, oyd)
        axs[2].plot(ox, oydd)

        leg.append("dmp reproduced")

        axs[0].plot(x, y,'-.')
        axs[1].plot(x, yd,'-.')
        axs[2].plot(x, ydd,'-.')
        leg.append("demonstrated")

        axs[0].plot(rx, ry, '--')
        axs[1].plot(rx, ryd, '--')
        axs[2].plot(rx, rydd, '--')
        leg.append("realised")

        plt.legend(leg)

        axs[0].set_xlabel("time [s]")
        axs[0].set_ylabel("y_" + str(i + 1))
        axs[1].set_xlabel("time [s]")
        axs[1].set_ylabel("yd_" + str(i + 1))
        axs[2].set_xlabel("time [s]")
        axs[2].set_ylabel("ydd_" + str(i + 1))

        if CONF_SAVEFIGS:
            if not os.path.exists("fig"):
                os.mkdir("fig")
            plt.savefig("fig/"+ PAR_FIG_NAMES + "_dmp" + str(i + 1) + ".eps")


# plotting demonstration and reproduced trajectories other then dmp data
def plot_data(dmp, dem_traj, x, xd, q, qd, qdd, ts=None, forcing_terms=None, fa_outputs=None, nonfiltered_traj=None):
    # collect data
    # reproduced trajectory

    dmp_repdata = dmp.statesAsTrajectory(ts, x, xd).asMatrix()

    # get data for plots
    dem_mdata = dem_traj.asMatrix()
    rep_mdata = np.column_stack((ts, q, qd, qdd))

    # plot trajectories
    if PAR_APPLY_LP_FILTER:
        plot_trajectories(dem_traj.dim(), dem_mdata, rep_mdata, nonfiltered_traj)
    else:
        plot_trajectories(dem_traj.dim(), dem_mdata, rep_mdata, dmp_repdata)

    plt.show()

def plot_phase(ts, phase):
    fig = plt.figure()

    plt.plot(ts,phase)
    plt.legend(["phase"])
    plt.xlabel("Time [s]")
    plt.ylabel("x")

    if CONF_SAVEFIGS:
        if not os.path.exists("fig"):
            os.mkdir("fig")
        plt.savefig("fig/" + PAR_FIG_NAMES + "_phase_evolution.eps")


#################################################################################################################
#   MAIN                                                                                                        #
#################################################################################################################
if __name__=="__main__":
    TIME_START  = time.time()
    # beginning of body of main

    # Init plant
    puma_robot_arm = PumaArm3DOF([PAR_G,PAR_THETA,PAR_M2,PAR_M3,PAR_L1,PAR_L2,PAR_L3])

    # Init controller
    if PAR_CONTROLLER_TYPE == "PID":
        controller = PIDController(3,[PAR_KP,PAR_KI,PAR_KD])
    elif PAR_CONTROLLER_TYPE == "FPT":
        controller = FPTController(3,[PAR_L,PAR_D,PAR_A],[PAR_aG,PAR_aTHETA,PAR_aM2,PAR_aM3,PAR_aL1,PAR_aL2,PAR_aL3])
    else:
        controller = None
        print("Selected controller is not supported! Please check the value for PAR_CONTROLLER_TYPE!")
        exit(1)



    # Parsing trajectory data from file and construct a dmp.Trajectory object
    raw_data = open(PAR_FILE_NAME, 'r').read()
    traj = prep_data(raw_data)
    # applying lp filter if configured
    if PAR_APPLY_LP_FILTER:
        nonfilt_traj = traj.asMatrix() # if I pass the traj it will be only a reference and not a new Trajectory object
        traj.applyLowPassFilter(PAR_CUTOFF_FREQ)
    else:
        nonfilt_traj = None

    function_apps = []
    l_fa_type = FUNC_APPROX_TYPES[PAR_FUNC_APPR_TYPE]
    if l_fa_type == "LWR":
        function_apps = [FunctionApproximatorLWR(PAR_NO_OF_BASIS) for x in range(traj.dim())]
    elif l_fa_type == "RBFN":
        function_apps = [FunctionApproximatorRBFN(PAR_NO_OF_BASIS) for x in range(traj.dim())]
    else:
        print("Selected function approximator type is not supported. Please check the value for PAR_FUNC_APPR_TYPE!")
        exit(1)

    # constructing the dmp objects
    dmp = Dmp(traj.dt_mean, traj.initial_y(), traj.final_y(), function_apps, name="Dmp", sigmoid_max_rate=PAR_SIGMOID_MR)

    # training DMP
    dmp.train(traj)

    # reproduce trajectory using the trained DMP
    forcing_terms = None
    fa_outputs = None
    (x, xd) = dmp.integrateStart()
    # T_f = (length(Qpath) + 1) * DMP.dt;               Qpath (number of elements of the path)
    # Xmin = exp(-DMP.a_x * T_f / DMP.tau);             a_x -> alpha_x
    # while phase > Xmin
    T_f = (traj.length()+1) * traj.dt_mean
    Xmin = math.exp((-2 * T_f)/dmp.tau_)
    phase = x[dmp.PHASE]

    # set init state to trajectory init
    init_state = puma_robot_arm.get_current_state()
    init_point_in_traj = np.array(traj.initial_y().reshape(dmp.dim_orig_,1))
    puma_robot_arm.init_state(init_point_in_traj,init_state[1],init_state[2])
    dt = traj.dt_mean


    # allocating memory
    ts = np.zeros(1)
    q_real = np.array(init_point_in_traj.transpose())
    qd_real = np.array(init_state[1].transpose())
    qdd_real = np.array(init_state[2].transpose())
    er = np.empty((1,dmp.dim_orig_))

    qdd_des_mem = np.empty((1,dmp.dim_orig_))
    qdd_def_mem = np.empty((1,dmp.dim_orig_))

    dmp_pos_error_mem = np.empty((1,dmp.dim_orig_))
    pos_error_mem = np.empty((1,dmp.dim_orig_))

    stucked_pos = np.zeros((1,dmp.dim_orig_))

    ## MAIN LOOP ##
    # while((1-phase) > Xmin):
    while(phase > Xmin):
        # integrate dmp
        # first iteration
        if len(np.shape(x)) == 1:
            (dmp_x_i, dmp_xd_i) = dmp.integrateStep(dt, x)
            ts_i = np.array([ts[-1]+dt])
        # second iteration and after ..
        else:
            # calculating dmp position error for \dy
            act_pos = np.array(q_real[-1,:])
            dmp_calc_pos = np.array(x[-1,dmp.SPRING_Y])

            # # artificial injection of scenario for phase stopping
            if CONF_INJECT_PHASE_STOP:
                step_i = np.shape(x)[0]
                if step_i == 150:
                    stucked_pos = np.array(act_pos)
                if (step_i >= 150) and (step_i <= 300):
                    act_pos = stucked_pos

            dmp_pos_error = np.array(act_pos-dmp_calc_pos)
            dmp.spring_system_.set_pos_error(dmp_pos_error)

            norm_dmp_pos_error = np.linalg.norm(dmp_pos_error)
            if norm_dmp_pos_error < PAR_ERROR_TRESHOLD:
                norm_dmp_pos_error = 0
            dmp.phase_system_.set_tracking_error(norm_dmp_pos_error)

            x_inp = np.array(x[-1,:])
            x_inp[dmp.SPRING_Y] = np.array(q_real[-1,:])
            (dmp_x_i, dmp_xd_i) = dmp.integrateStep(dt, x_inp)
            ts_i = np.array([ts[-1,-1]+dt])


        # gathering input for the controller (calculate error terms, feedforward term)
        current_pos = np.array(puma_robot_arm.get_current_state()[0])
        ref_pos = np.array(dmp_x_i[dmp.SPRING_Y]).reshape((dmp.dim_orig_, 1))

        current_dpos = np.array(puma_robot_arm.get_current_state()[1])
        ref_dpos = np.array(dmp_xd_i[dmp.SPRING_Y]).reshape((dmp.dim_orig_, 1))

        qdd_ref = np.array(dmp_xd_i[dmp.SPRING_Z]).reshape((dmp.dim_orig_, 1))

        # calculating error for controller
        error = ref_pos - current_pos
        d_error = ref_dpos - current_dpos

        # executing controller
        if PAR_CONTROLLER_TYPE == "FPT":
            [des_ddq, def_ddq, actuating_forces] = controller.calc_action_forces(qdd_ref, error, d_error, dt)
            qdd_def_mem = np.row_stack((qdd_def_mem,def_ddq.transpose()))
            qdd_des_mem = np.row_stack((qdd_des_mem,des_ddq.transpose()))
        else:  # PAR_CONTROLLER_TYPE == "PID":
            actuating_forces = controller.calc_action_forces(qdd_ref, error, d_error, dt)

        # feed the action forces to the robot
        ddq_feedback = puma_robot_arm.calc_effect_of_action_forces(actuating_forces)

        # update robot state using euler integration
        puma_robot_arm.update_state(ddq_feedback, dt)

        if PAR_CONTROLLER_TYPE == "FPT":
            # updating realised effect for the controller
	        controller.update_real_effect(ddq_feedback)


        # store values for plotting
        x = np.row_stack((x,dmp_x_i))
        xd = np.row_stack((xd,dmp_xd_i))
        ts = np.row_stack((ts,ts_i))

        st = puma_robot_arm.get_current_state()
        q_real = np.row_stack((q_real,st[0].transpose()))
        qd_real = np.row_stack((qd_real,st[1].transpose()))
        qdd_real = np.row_stack((qdd_real,st[2].transpose()))
        er = np.row_stack((er,error.transpose()))

        # update phase value for condition
        phase = x[-1,dmp.PHASE]

    ## END OF MAIN LOOP

    # plot data
    plot_phase(ts, x[:, dmp.PHASE])
    plot_data(dmp, traj, x, xd, q_real, qd_real, qdd_real, ts, forcing_terms, fa_outputs, nonfiltered_traj=nonfilt_traj)


    # end of body of main
    TIME_END    = time.time()
    # print execution time
    print("--- %s seconds ---" % (TIME_END - TIME_START))
    pass
#################################################################################################################
#   UNIT TESTS                                                                                                  #
#################################################################################################################