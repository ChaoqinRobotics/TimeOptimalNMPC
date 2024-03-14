import numpy as np
import casadi as ca
from casadi import MX, DM, vertcat, mtimes, Function, inv, cross, sqrt, norm_2

import yaml

# Quaternion Multiplication
def quat_mult(q1,q2):
    ans = ca.vertcat(q2[0,:] * q1[0,:] - q2[1,:] * q1[1,:] - q2[2,:] * q1[2,:] - q2[3,:] * q1[3,:],
           q2[0,:] * q1[1,:] + q2[1,:] * q1[0,:] - q2[2,:] * q1[3,:] + q2[3,:] * q1[2,:],
           q2[0,:] * q1[2,:] + q2[2,:] * q1[0,:] + q2[1,:] * q1[3,:] - q2[3,:] * q1[1,:],
           q2[0,:] * q1[3,:] - q2[1,:] * q1[2,:] + q2[2,:] * q1[1,:] + q2[3,:] * q1[0,:])
    return ans

# Quaternion-Vector Rotation
def rotate_quat(q1,v1):
    ans = quat_mult(quat_mult(q1, ca.vertcat(0, v1)), ca.vertcat(q1[0,:],-q1[1,:], -q1[2,:], -q1[3,:]))
    return ca.vertcat(ans[1,:], ans[2,:], ans[3,:]) # to covert to 3x1 vec

def RK4(f_c:ca.Function, X0, U, dt, M:int):
    DT = dt/M
    X1 = X0
    for _ in range(M):
        k1 = DT*f_c(X1,        U)
        k2 = DT*f_c(X1+0.5*k1, U)
        k3 = DT*f_c(X1+0.5*k2, U)
        k4 = DT*f_c(X1+k3,     U)
        X1 = X1+(k1+2*k2+2*k3+k4)/6
    # F = ca.Function('F', [X0, U], [X1] ,['X0', 'U'], ['X1'])
    return X1

def EulerIntegral(f_c:ca.Function, X0, U, dt, M:int):
    DT = dt/M
    X1 = X0
    for _ in range(M):
        X1 = X1 + DT*f_c(X1, U)
    
    return X1

def constrain(a, lb, ub):
    if a<lb:
        a=lb
    if a>ub:
        a=ub
    return a

class QuadrotorModel(object):
    def __init__(self, cfg_f):
        
        self._m = 1.0         # total mass
        self._arm_l = 0.23    # arm length
        self._beta = np.pi / 4
        self._has_beta = False
        self._c_tau = 0.0133  # torque constant
        
        self._G = 9.8066
        self._J = np.diag([0.01, 0.01, 0.02])     # inertia
        self._J_inv = np.linalg.inv(self._J)
        self._D = np.diag([0.6, 0.6, 0.6])
        
        self._v_xy_max = ca.inf
        self._v_z_max = ca.inf
        self._omega_xy_max = 5
        self._omega_z_max = 1
        self._T_max = 4.179
        self._T_min = 0

        self.load(cfg_f)
        
        self._X_lb = [-ca.inf, -ca.inf, -ca.inf,
                      -self._v_xy_max, -self._v_xy_max, -self._v_z_max,
                      -1,-1,-1,-1,
                      -self._omega_xy_max, -self._omega_xy_max, -self._omega_z_max]
        self._X_ub = [ca.inf, ca.inf, ca.inf,
                      self._v_xy_max, self._v_xy_max, self._v_z_max,
                      1,1,1,1,
                      self._omega_xy_max, self._omega_xy_max, self._omega_z_max]

        self._U_lb = [self._T_min, self._T_min, self._T_min, self._T_min]
        self._U_ub = [self._T_max, self._T_max, self._T_max, self._T_max]

    def load(self, cfg_f):
      with open(cfg_f, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        if "mass" in cfg:
          self._m = cfg["mass"]
        else:
          print("No mass specified in " + cfg_f)
        
        if "arm_length" in cfg:
          self._arm_l = cfg["arm_length"]
        else:
          print("No arm length specified in " + cfg_f)

        if "torCoeff" in cfg:
          self._c_tau = cfg["torCoeff"]
        else:
          print("No torque coefficient specified in " + cfg_f)
        
        if "beta" in cfg:
          self._beta = cfg["beta"]
          self._has_beta = True
        else:
          print("Use default drone frame configuration since no beta specified in " + cfg_f)
          self._has_beta = False

        self._has_tbm = False
        if "tbm_fr" in cfg and "tbm_bl" in cfg and "tbm_br" in cfg and "tbm_fl" in cfg:
          self._tbm_fr = cfg["tbm_fr"]
          self._tbm_bl = cfg["tbm_bl"]      
          self._tbm_br = cfg["tbm_br"]
          self._tbm_fl = cfg["tbm_fl"]
          self._tbm = np.matrix([self._tbm_fr, self._tbm_bl, self._tbm_br, self._tbm_fl]).T
          self._has_tbm = True
          print("Thrust mixing matrix: \n", self._tbm)

        if "inertia" in cfg:
          self._J = np.diag(cfg["inertia"])
          self._J_inv = np.linalg.inv(self._J)
        else:
          print("No inertia specified in " + cfg_f)
        
        if "omega_max" in cfg:
          omega_max = np.array(cfg["omega_max"])
          self._omega_xy_max = omega_max[0]
          self._omega_z_max = omega_max[2]
        else:
          print("No omega_max specified in " + cfg_f)
        
        if "thrust_min" in cfg:
          self._T_min = cfg["thrust_min"]
        else:
          print("No min thrust specified in " + cfg_f)
        if "thrust_max" in cfg:
          self._T_max = cfg["thrust_max"]
        else:
          print("No max thrust specified in " +cfg_f)
    

    def dynamics(self):
      p = ca.MX.sym('p', 3)
      v = ca.MX.sym('v', 3)
      q = ca.MX.sym('q', 4)
      w = ca.MX.sym('w', 3)
      T = ca.MX.sym('thrust', 4)

      x = vertcat(p, v, q, w)
      u = vertcat(T)

      g = DM([0, 0, -self._G])

      x_dot = []

      if self._has_tbm:
        tbm = self._tbm
        x_dot = vertcat(
          v,
          rotate_quat(q, vertcat(0, 0, (T[0]+T[1]+T[2]+T[3])/self._m)) + g,
          0.5*quat_mult(q, vertcat(0, w)),
          mtimes(self._J_inv, vertcat(
            (tbm.item((0, 0))*T[0]+tbm.item((0, 1))*T[1]+tbm.item((0, 2))*T[2]+tbm.item((0, 3))*T[3]),
            (tbm.item((1, 0))*T[0]+tbm.item((1, 1))*T[1]+tbm.item((1, 2))*T[2]+tbm.item((1, 3))*T[3]),
            self._c_tau*(-T[0]-T[1]+T[2]+T[3]))
          -cross(w,mtimes(self._J,w)))
        )

        M = np.array([[1, 1, 1, 1], self._tbm])
        print("Thrust mixing matrix: \n", M)  

      elif self._has_beta:
        lsb = self._arm_l * np.sin(self._beta * np.pi / 180.0)
        lcb = self._arm_l * np.cos(self._beta * np.pi  / 180.0)
        x_dot = vertcat(
          v,
          rotate_quat(q, vertcat(0, 0, (T[0]+T[1]+T[2]+T[3])/self._m)) + g,
          0.5*quat_mult(q, vertcat(0, w)),
          mtimes(self._J_inv, vertcat(
            (-lsb*T[0]+lsb*T[1]-lsb*T[2]+lsb*T[3]),
            (-lcb*T[0]+lcb*T[1]+lcb*T[2]-lcb*T[3]),
            self._c_tau*(-T[0]-T[1]+T[2]+T[3]))
          -cross(w,mtimes(self._J,w)))
        )

        M = np.array([[1, 1, 1, 1], [-lsb, lsb, -lsb, lsb], [-lcb, lcb, lcb, -lcb], [-self._c_tau, -self._c_tau, self._c_tau, self._c_tau]])
        print("Thrust mixing matrix: \n", M)
      else:
        print("Default drone frame configuration used")
        x_dot = vertcat(
          v,
          rotate_quat(q, vertcat(0, 0, (T[0]+T[1]+T[2]+T[3])/self._m)) + g,
          0.5*quat_mult(q, vertcat(0, w)),
          mtimes(self._J_inv, vertcat(
            self._arm_l*(T[0]-T[1]-T[2]+T[3]),
            self._arm_l*(-T[0]-T[1]+T[2]+T[3]),
            self._c_tau*(T[0]-T[1]+T[2]-T[3]))
          -cross(w,mtimes(self._J,w)))
        )
        # M = np.array([[1, 1, 1, 1], self._arm_l*[1, -1, -1, 1], self._arm_l*[-1, -1, 1, 1], self._c_tau*[1, -1, 1, -1]])
        # print("Thrust mixing matrix: \n", M)

      fx = Function('f',  [x, u], [x_dot], ['x', 'u'], ['x_dot'])
      return fx

    # dt is specified
    def ddynamics(self, dt):
        f = self.dynamics()
        X0 = ca.SX.sym("X", f.size1_in(0))
        U = ca.SX.sym("U", f.size1_in(1))
        
        X1 = RK4(f, X0, U, dt, 1)
        # X1 = EulerIntegral(f, X0, U, dt, 1)
        q_l = ca.sqrt(X1[6:10].T@X1[6:10])
        X1[6:10] = X1[6:10]/q_l
        
        return ca.Function("ddyn", [X0, U], [X1], ["X0", "U"], ["X1"])
    
    # dt is an variable
    def ddynamics_dt(self):
        f = self.dynamics()
        X0 = ca.SX.sym("X", f.size1_in(0))
        U = ca.SX.sym("U", f.size1_in(1))
        dt = ca.SX.sym('dt')
        # TODO: us RK4 integration
        X1 = RK4(f, X0, U, dt, 1)
        # X1 = EulerIntegral(f, X0, U, dt, 1)
        q_l = ca.sqrt(X1[6:10].T@X1[6:10])
        X1[6:10] = X1[6:10]/q_l
        # State prediction
        return ca.Function("ddyn_t", [X0, U, dt], [X1], ["X0", "U", "dt"], ["X1"])

