import numpy as np
import casadi as ca

from quadrotor import QuadrotorModel
from nmpc_params import NmpcParams

def calc_Ns(params:NmpcParams, xinit, xend):
    Ns = []
    l_per_n = params._sample_dist

    if len(params._waypoints) == 0:
      l = np.linalg.norm(np.array(xend[:3])-np.array(xinit[:3]))
      Ns.append(int(l/l_per_n))
      return Ns

    # Samples to reach the first gate
    l = np.linalg.norm(np.array(params._waypoints[0])-np.array(xinit[:3]))
    Ns.append(int(l/l_per_n))
    # Samples for each trajectory segment
    for i in range(params._wpt_num - 1):
        l = np.linalg.norm(np.array(params._waypoints[i])-np.array(params._waypoints[i+1]))
        Ns.append(int(l/l_per_n))
    # The last gate's position is the terminal position
    l = np.linalg.norm(xend[:3] - np.array(params._waypoints[-1]))
    Ns.append(int(l/l_per_n)) 
    return Ns

class Nmpc():
    def __init__(self, quad:QuadrotorModel, params:NmpcParams, xinit, xend):
        # Quadrotor dynamics
        self._quad = quad
        self._ddynamics = self._quad.ddynamics_dt()
        self._params = params

        self._tol = params._tol_wpt
        self._tol_term = params._tol_term

        # Number of gates
        self._wp_num = params._wpt_num

        self._Ns = calc_Ns(params, xinit, xend)
        self._seg_num = len(self._Ns)
        assert(self._seg_num==self._wp_num + 1)
        self._Horizon = 0
        self._N_wp_base = [0]
        # _N_wp_base: index to each segment
        # _Horizon: total sampled nodes
        for i in range(self._seg_num):
            self._N_wp_base.append(self._N_wp_base[i]+self._Ns[i])
            self._Horizon += self._Ns[i]
        print("Total points: ", self._Horizon)
        # Dimension of states
        self._X_dim = self._ddynamics.size1_in(0)
        # Dimension of controls
        self._U_dim = self._ddynamics.size1_in(1)
        # Lower and upper bounds
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub

        # Time optimization variables
        self._DTs = ca.SX.sym('DTs', self._seg_num)
        # State optimization variables
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Horizon)
        # Control optimization variables
        self._Us = ca.SX.sym('Us', self._U_dim, self._Horizon)
        # Parameters for waypoints
        self._WPs_p = ca.SX.sym('WPs_p', 3, self._wp_num)

        # Initialization values
        self._X_init = ca.SX.sym('X_init', self._X_dim)
        self._X_end = ca.SX.sym('X_end', self._X_dim)

        # Weights for bodyrate regulation
        self._cost_Co = ca.diag([0.01,0.01,0.01]) # opt param
        # Weights for waypoint-constraint violation
        self._cost_WP_p = ca.diag([1,1,1]) # opt param
        self._cost_state = ca.diag([1,1,1,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.05,0.05,0.05]) # opt param

        self._opt_option = {
            # 'verbose': False,
            'ipopt.tol': 1e-5,
            # 'ipopt.acceptable_tol': 1e-3,
            'ipopt.max_iter': 1000,
            # 'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 0,
        }

        self._opt_t_option = {
            'verbose': False,
            # 'ipopt.tol': 1e-2,
            # 'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0
        }

        self._opt_t_warm_option = {
            'verbose': False,
            # 'ipopt.tol': 1e-2,
            # 'ipopt.acceptable_tol': 1e-2,
            'ipopt.max_iter': 1000,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_frac': 1e-6,
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_frac': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
            'ipopt.print_level': 0
        }

        #################################################################
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []

        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []

        self._nlp_x_t = []
        self._nlp_lbx_t = []
        self._nlp_ubx_t = []

        self._nlp_g_orientation = []
        self._nlp_lbg_orientation = []
        self._nlp_ubg_orientation = []

        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []

        self._nlp_g_wp_p = []
        self._nlp_lbg_wp_p = []
        self._nlp_ubg_wp_p = []

        self._nlp_g_quat = []
        self._nlp_lbg_quat = []
        self._nlp_ubg_quat = []


        self._nlp_p_xinit = [ self._X_init ]
        self._xinit = xinit
        self._nlp_p_xend = [ self._X_end ]
        self._xend = xend

        self._nlp_p_dt = []
        self._nlp_p_wp_p = []
        
        self._nlp_obj_orientation = 0
        self._nlp_obj_minco = 0
        self._nlp_obj_time = 0
        self._nlp_obj_wp_p = 0
        self._nlp_obj_quat = 0
        self._nlp_obj_dyn = 0

        ###################################################################

        for i in range(self._seg_num):
            # Add the first state variables and constraints for segment i
            self._nlp_x_x += [ self._Xs[:, self._N_wp_base[i]] ]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            # Add the first control variables and constraints for segment i
            self._nlp_x_u += [ self._Us[:, self._N_wp_base[i]] ]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            # Add time varables and constraints in durations
            self._nlp_x_t += [ self._DTs[i] ]
            self._nlp_lbx_t += [0]
            self._nlp_ubx_t += [0.5]

            # Dynamic constraint as cost functions (for the first point)
            if i==0:
                # self._Xs[:,0]: initial state
                # self._ddynamics( self._X_init, self._Us[:,0], self._DTs[0]): prediction of the next state at dt(0)
                # Note: the initial state is not treated as variable
                dd_dyn = self._Xs[:,0]-self._ddynamics( self._X_init, self._Us[:,0], self._DTs[0])
                self._nlp_g_dyn += [ dd_dyn ] # Equality constriants
                # matrix multiplication, A @ B
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn # Norm of the result, for cost function?
                self._nlp_obj_minco += self._Xs[10:13,0].T@self._cost_Co@self._Xs[10:13,0] # Bodyrate regulation term
            else:
                dd_dyn = self._Xs[:,self._N_wp_base[i]]-self._ddynamics( self._Xs[:,self._N_wp_base[i]-1], self._Us[:,self._N_wp_base[i]], self._DTs[i])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
                self._nlp_obj_minco += self._Xs[10:13,self._N_wp_base[i]].T@self._cost_Co@self._Xs[10:13,self._N_wp_base[i]]

            # Set zero bound for dynamics constraints
            self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
            self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]
            
            # Waypoint constraints
            # The last point in each segment should match the desired waypoint
            # This one is used as hard constraint
            # self._nlp_g_wp_p += [ (self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]).T@(self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]) ]
            # self._nlp_lbg_wp_p += [0]
            # self._nlp_ubg_wp_p += [ self._tol*self._tol ] # A tolerance of 0.01 is set by default

            if i==self._seg_num-1:
              # The last point
              self._nlp_g_wp_p += [ (self._Xs[:,self._N_wp_base[i+1]-1]-self._X_end).T@(self._Xs[:,self._N_wp_base[i+1]-1]-self._X_end) ]
              self._nlp_lbg_wp_p += [0]
              self._nlp_ubg_wp_p += [ self._tol_term*self._tol_term ] # A tolerance of 0.01 is set by default
              self._nlp_obj_wp_p += (self._Xs[:,self._N_wp_base[i+1]-1]-self._X_end).T@self._cost_state@(self._Xs[:,self._N_wp_base[i+1]-1]-self._X_end)
            else:
              self._nlp_g_wp_p += [ (self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]).T@(self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]) ]
              self._nlp_lbg_wp_p += [0]
              self._nlp_ubg_wp_p += [ self._tol*self._tol ] # A tolerance of 0.01 is set by default
              self._nlp_obj_wp_p += (self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i]).T@self._cost_WP_p@(self._Xs[:3,self._N_wp_base[i+1]-1]-self._WPs_p[:,i])
              self._nlp_p_wp_p += [ self._WPs_p[:,i] ]
            # Add time gaps and waypoint positions as parameters
            self._nlp_p_dt += [ self._DTs[i] ]
            

            # self._nlp_obj_minco += (self._Us[:,self._N_wp_base[i]]).T@self._cost_Co@(self._Us[:,self._N_wp_base[i]])

            # Total trajectory time used as objectives
            self._nlp_obj_time += self._DTs[i]*self._Ns[i]
            # Waypoint constraints as objectives
            
            # Add variables and constraitns for the rest of the state
            for j in range(1, self._Ns[i]):
                self._nlp_x_x += [ self._Xs[:, self._N_wp_base[i]+j] ]
                self._nlp_lbx_x += self._X_lb
                self._nlp_ubx_x += self._X_ub
                self._nlp_x_u += [ self._Us[:, self._N_wp_base[i]+j] ]
                self._nlp_lbx_u += self._U_lb
                self._nlp_ubx_u += self._U_ub
                
                dd_dyn = self._Xs[:,self._N_wp_base[i]+j]-self._ddynamics( self._Xs[:,self._N_wp_base[i]+j-1], self._Us[:,self._N_wp_base[i]+j], self._DTs[i])
                self._nlp_g_dyn += [ dd_dyn ]
                self._nlp_obj_dyn += dd_dyn.T@dd_dyn
                self._nlp_lbg_dyn += [ -0.0 for _ in range(self._X_dim) ]
                self._nlp_ubg_dyn += [  0.0 for _ in range(self._X_dim) ]

                # self._nlp_obj_minco += (self._Us[:,self._N_wp_base[i]+j]).T@self._cost_Co@(self._Us[:,self._N_wp_base[i]+j])
                self._nlp_obj_minco += self._Xs[10:13,self._N_wp_base[i]+j].T@self._cost_Co@self._Xs[10:13,self._N_wp_base[i]+j]

    # Configure the warm-up solver
    def define_opt(self):
        # OCP without dynamics constraints
        nlp_dect = {
            'f': 1*self._nlp_obj_dyn + self._nlp_obj_wp_p + 1*self._nlp_obj_minco, # 1.0 is the weight
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_xend+self._nlp_p_wp_p+self._nlp_p_dt)),
            # 'g': ca.vertcat(*(self._nlp_g_dyn)),
        }

        # Configure the solver
        self._opt_solver = ca.nlpsol('opt', 'ipopt', nlp_dect, self._opt_option)
        # Initialil values for states and controls (except for quaternion.w and thrust)
        self._xu0 = np.zeros((self._X_dim+self._U_dim)*self._Horizon)
        for i in range(self._Horizon):
            self._xu0[i*self._X_dim+6] = 1 # Quaternion
            # TODO: why only assign -9.8066 to the first thrust?
            # self._xu0[self._Horizon*self._X_dim+i*self._U_dim] = -9.8066 
        self._xu0[self._Horizon*self._X_dim:] = 9.8066/4

    # Solve the warm-up problem
    def solve_opt(self, dts):

        p = np.zeros(2*self._X_dim+3*self._wp_num+self._seg_num)
        p[:self._X_dim] = self._xinit
        p[self._X_dim : 2*self._X_dim] = self._xend
        p[2*self._X_dim : 2*self._X_dim+3*self._wp_num] = np.array(self._params._waypoints).flatten()
        p[2*self._X_dim+3*self._wp_num : ] = dts

        # Solve the OCP; only state and input constraints are imposed
        res = self._opt_solver(
            x0=self._xu0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u),
            # lbg=(),
            # ubg=(),
            p=p
        )
        self._xu0 = res['x'].full().flatten()
        self._dt0 = dts
        # Initialize the optimization variables: self._xut0
        self._xut0 = np.zeros((self._X_dim+self._U_dim)*self._Horizon+self._seg_num)
        self._xut0[:(self._X_dim+self._U_dim)*self._Horizon] = self._xu0
        self._xut0[(self._X_dim+self._U_dim)*self._Horizon:] = self._dt0
        return res

    # time-optimal planner solver
    def define_opt_t(self):
        # Setup the problem with dynamics constraints
        nlp_dect = {
            'f': self._nlp_obj_time,
            'x': ca.vertcat(*(self._nlp_x_x+self._nlp_x_u+self._nlp_x_t)),
            'p': ca.vertcat(*(self._nlp_p_xinit+self._nlp_p_xend+self._nlp_p_wp_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn+self._nlp_g_wp_p)),
        }
        # Warm-up or not depends on the optimization parameters
        self._opt_t_solver = ca.nlpsol('opt_t', 'ipopt', nlp_dect, self._opt_t_option)
        self._lam_x0 = np.zeros(self._opt_t_solver.size_in(6)[0])
        self._lam_g0 = np.zeros(self._opt_t_solver.size_in(7)[0])
        
    def solve_opt_t(self):
        p = np.zeros(2*self._X_dim+3*self._wp_num)
        p[:self._X_dim] = self._xinit
        p[self._X_dim:2*self._X_dim] = self._xend
        p[2*self._X_dim:2*self._X_dim+3*self._wp_num] = np.array(self._params._waypoints).flatten()

        # The only difference is in parameter settings
        # Use self._xut0 as initual guesses
        res = self._opt_t_solver(
            x0=self._xut0,
            lam_x0 = self._lam_x0,
            lam_g0 = self._lam_g0,
            lbx=(self._nlp_lbx_x+self._nlp_lbx_u+self._nlp_lbx_t),
            ubx=(self._nlp_ubx_x+self._nlp_ubx_u+self._nlp_ubx_t),
            lbg=(self._nlp_lbg_dyn+self._nlp_lbg_wp_p),
            ubg=(self._nlp_ubg_dyn+self._nlp_ubg_wp_p),
            p=p
        )
        # Get the optimized variables 
        self._xut0 = res['x'].full().flatten()
        # Lagrangian for constriants
        self._lam_x0 = res["lam_x"]
        self._lam_g0 = res["lam_g"]

        dts = self._xut0[-self._seg_num:]
        print("optimized dts: ", dts)
        return res
