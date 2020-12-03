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

#################################################################################################################
#   DEFAULT CONFIG                                                                                              #
#################################################################################################################
CONF_DEBUG                              = False
CONF_SAVEFIGS                           = False
CONF_PLOT_DMP_WITH_LIBPROVIDED_FUNCTION = False
CONF_PRINT_NONFILTERED_DATA             = False
CONF_ANALYTICAL_SOL                     = False
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
PAR_FILE_NAME               = 'trajectory/data/matlab_generated_polynomials.txt'   #panda_pick_and_place_7dof_trajectory.txt
PAR_SIGMOID_MR              = -1    # value for max rate of change within the sigmoid dynamical system
PAR_FUNC_APPR_TYPE          = 0     # 0 for LWR and 1 for RBFN method
PAR_NO_OF_BASIS             = 12     # number of basis functions used for the function approximators
PAR_APPLY_LP_FILTER         = False  # applying low-pass filter on demonstration data
PAR_CUTOFF_FREQ             = 0.9  # cutoff frequency used for the LP filter
PAR_FIG_NAMES               = "phase_stopping2dof"
PAR_ERROR_TRESHOLD          = 0.004


#TODO implement phase stopping without the system (1 or 2 DOF)
#TODO if phase stopping works integrate everything together again, (stop the PUMA robot)

#TODO if I receive the MatLab code, get familiar with it, but don't start to refactor etc..
#TODO get more familiar with quaternions


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
        if CONF_PRINT_NONFILTERED_DATA:
            if PAR_APPLY_LP_FILTER:
                ox   = orig_mdata[:, 0]
                oy   = orig_mdata[:, i+1]
                oyd  = orig_mdata[:, dim + i+1]
                oydd = orig_mdata[:, (2 * dim) + i+1]

                axs[0].plot(ox, oy)
                axs[1].plot(ox, oyd)
                axs[2].plot(ox, oydd)

                leg.append("original non-filtered trajectory")



        axs[0].plot(x, y)
        axs[1].plot(x, yd)
        axs[2].plot(x, ydd)
        leg.append("demonstrated")

        axs[0].plot(rx, ry, '--')
        axs[1].plot(rx, ryd, '--')
        axs[2].plot(rx, rydd, '--')
        leg.append("reproduced")

        plt.legend(leg)

        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("y_" + str(i + 1))
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("yd_" + str(i + 1))
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("ydd_" + str(i + 1))

        if CONF_SAVEFIGS:
            if not os.path.exists("fig"):
                os.mkdir("fig")
            plt.savefig("fig/"+ PAR_FIG_NAMES + "_dmp" + str(i + 1) + ".eps")

        # fige = plt.figure()
        #
        # e_x = x[:len(rx)]
        # e_y = y[:len(ry)]-ry
        # e_yd = yd[:len(ryd)] - ryd
        # e_ydd = ydd[:len(rydd)] - rydd
        # axse = [fige.add_subplot(311), fige.add_subplot(312), fige.add_subplot(313)]
        # axse[0].plot(e_x,e_y)
        # axse[1].plot(e_x,e_yd)
        # axse[2].plot(e_x,e_ydd)




# plotting demonstration and reproduced trajectories other then dmp data
def plot_data(dmp, dem_traj, xs, xds, ts=None, forcing_terms=None, fa_outputs=None, nonfiltered_traj=None):
    # collect data
    # reproduced trajectory
    if CONF_ANALYTICAL_SOL:
        traj_reproduced = dmp.statesAsTrajectory(dmp.ts_train_, xs, xds)
    else:
        traj_reproduced = dmp.statesAsTrajectory(ts,xs,xds)

    # get data for plots
    dem_mdata = dem_traj.asMatrix()
    rep_mdata = traj_reproduced.asMatrix()

    # plot trajectories
    if PAR_APPLY_LP_FILTER:
        plot_trajectories(dem_traj.dim(), dem_mdata, rep_mdata, nonfiltered_traj)
    else:
        plot_trajectories(dem_traj.dim(), dem_mdata, rep_mdata)


    # plot DMP
    if CONF_PLOT_DMP_WITH_LIBPROVIDED_FUNCTION:
        fig = plt.figure()
        plotDmp(np.column_stack((dmp.ts_train_,xs,xds)),fig, forcing_terms, fa_outputs)



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
    if CONF_ANALYTICAL_SOL:
        (x, xd, forcing_terms, fa_outputs) = dmp.analyticalSolution()
        ts = None
    else:
        ts = np.zeros(1)
        forcing_terms = None
        fa_outputs = None
        (x, xd) = dmp.integrateStart()
        # T_f = (length(Qpath) + 1) * DMP.dt;               Qpath (number of elements of the path)
        # Xmin = exp(-DMP.a_x * T_f / DMP.tau);             a_x -> alpha_x
        # while phase > Xmin
        T_f = (traj.length()+1) * traj.dt_mean
        Xmin = math.exp((-2 * T_f)/dmp.tau_)
        phase = x[dmp.PHASE]

        stucked_pos = np.zeros((1,dmp.dim_orig_))

        #while((1-phase) > Xmin):
        while ((phase) > Xmin):
            # integrate dmp

            # first iteration
            if len(np.shape(x)) == 1:
                (x_i, xd_i) = dmp.integrateStep(traj.dt_mean, x)
                ts_i = np.array([ts[-1]+traj.dt_mean])
            # when x is more then one row
            else:
                x_inp = np.array(x[-1,:])
                # # artificial injection of scenario for phase stopping
                act_pos = np.array(x_inp[dmp.SPRING_Y])
                if CONF_INJECT_PHASE_STOP:
                    step_i = np.shape(x)[0]
                    if step_i == 150:
                        stucked_pos = x[-1,dmp.SPRING_Y]
                    if (step_i >= 150) and (step_i <= 300):
                        act_pos = stucked_pos
                dmp_calc_pos = np.array(x[-1, dmp.SPRING_Y])
                dmp_pos_error = np.array(act_pos - dmp_calc_pos)
                dmp.spring_system_.set_pos_error(dmp_pos_error)

                norm_dmp_pos_error = np.linalg.norm(dmp_pos_error)
                if norm_dmp_pos_error < PAR_ERROR_TRESHOLD:
                    norm_dmp_pos_error = 0
                dmp.phase_system_.set_tracking_error(norm_dmp_pos_error)


                (x_i, xd_i) = dmp.integrateStep(traj.dt_mean, x_inp)
                ts_i = np.array([ts[-1,-1]+traj.dt_mean])

            # store current state to memory
            ts = np.row_stack((ts,ts_i))

            x = np.row_stack((x,x_i))
            xd = np.row_stack((xd,xd_i))
            # update phase value for condition
            phase = x[-1,dmp.PHASE]


    # plot data
    plot_phase(ts, x[:,dmp.PHASE])
    plot_data(dmp, traj, x, xd, ts, forcing_terms, fa_outputs, nonfiltered_traj=nonfilt_traj)


    # end of body of main
    TIME_END    = time.time()
    # print execution time
    print("--- %s seconds ---" % (TIME_END - TIME_START))
    pass
#################################################################################################################
#   UNIT TESTS                                                                                                  #
#################################################################################################################