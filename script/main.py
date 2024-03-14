import numpy as np
import csv

import sys, os
BASEPATH = os.path.abspath(__file__).split("script/", 1)[0]
sys.path += [BASEPATH]

from quadrotor import QuadrotorModel
from nmpc import Nmpc
from nmpc_params import NmpcParams

class Optimization():
    def __init__(self, quad:QuadrotorModel, params:NmpcParams):
        self._quad = quad
        self._params = params

    def solve(self, xinit, xend):
        nmpc = Nmpc(quad, params, xinit, xend)

        dts = np.array([0.3]*(params._wpt_num + 1))

        print("\n\nWarm-up start ......\n")
        nmpc.define_opt()
        res = nmpc.solve_opt(dts)

        print("\n\nTime optimization start ......\n")
        nmpc.define_opt_t()
        res_t = nmpc.solve_opt_t()
        # Save the trajectory inside the results folder
        self.save_traj(res_t, nmpc, BASEPATH+"/results/traj.csv")    

    def save_traj(self, res, ctr: Nmpc, csv_f):
        with open(csv_f, 'w') as f:
            traj_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #TODO: modify it to match the agilicious format csv
            labels = ['t',
                    "p_x", "p_y", "p_z",
                    "v_x", "v_y", "v_z",
                    "q_w", "q_x", "q_y", "q_z",
                    "w_x", "w_y", "w_z",
                    "a_lin_x", "a_lin_y", "a_lin_z",
                    "a_rot_x", "a_rot_y", "a_rot_z",
                    "u_1", "u_2", "u_3", "u_4",
                    "jerk_x", "jerk_y", "jerk_z",
                    "snap_x", "snap_y", "snap_z"]
            traj_writer.writerow(labels)
            x = res['x'].full().flatten()
            
            t = 0
            s = ctr._xinit
            u = x[ctr._Horizon*ctr._X_dim: ctr._Horizon*ctr._X_dim+ctr._U_dim]
            u_last = u

            ###
            a_lin = [0,0,0]
            a_rot = [0,0,0]
            jerk = [0,0,0]
            snap = [0,0,0]

            traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], a_lin[0], a_lin[1], a_lin[2], a_rot[0], a_rot[1], a_rot[2], u[0], u[1], u[2], u[3], jerk[0], jerk[1], jerk[2], snap[0], snap[1], snap[2]])
            for i in range(ctr._seg_num):
                # ctrimized time gap
                dt = x[-(ctr._seg_num)+i]
                for j in range(ctr._Ns[i]):
                    idx = ctr._N_wp_base[i]+j
                    t += dt
                    s = x[idx*ctr._X_dim: (idx+1)*ctr._X_dim]
                    if idx != ctr._Horizon-1:
                        u = x[ctr._Horizon*ctr._X_dim+(idx+1)*ctr._U_dim: ctr._Horizon*ctr._X_dim+(idx+2)*ctr._U_dim]
                        u_last = u
                    else:
                        u = u_last
                    traj_writer.writerow([t, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], a_lin[0], a_lin[1], a_lin[2], a_rot[0], a_rot[1], a_rot[2], u[0], u[1], u[2], u[3], jerk[0], jerk[1], jerk[2], snap[0], snap[1], snap[2]])
            print("--------------------------")
            print("Minimum lap time: ", t)

if __name__ == "__main__": 
    # Load quadrotor model parameters
    quad = QuadrotorModel(BASEPATH+'/parameters/quad_params.yaml')
    # Load NMPC optimization parameters
    params = NmpcParams(BASEPATH+'/parameters/ctr_params.yaml')
    # Instantiate an NMPC planner
    optimization = Optimization(quad, params)

    # Set the initial state [position, velocity, quaternion (rotation), bodyrate]
    xinit = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0])
    # Set the target state [position, velocity, quaternion (rotation), bodyrate]
    xend = np.array([15,0,0, 0,0,0, 1,0,0,0, 0,0,0])
    optimization.solve(xinit, xend)





