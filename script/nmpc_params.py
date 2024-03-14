import numpy as np
import yaml

class NmpcParams():
    def __init__(self, yaml_f=""):
        
        self._waypoints = []
        self._wpt_num = 0
        self._tol_wpt = 0
        self._tol_term = 0
        self._sample_dist = 0.3
        self._Ns = []

        if yaml_f!="":
            self.load_from(yaml_f)

    def load_from(self, yaml_f):
        with open(yaml_f, 'r') as f:
            gf = yaml.load(f, Loader=yaml.FullLoader)
            self._tol_wpt = gf["tol_wpt"]
            self._tol_term = gf["tol_term"]   
            self._sample_dist = gf["sample_dist"]    
            self._waypoints = gf["waypoints"]
        self._wpt_num = len(self._waypoints)
        # self._Ns = self.calc_Ns()

    # def calc_Ns(self):
    #     Ns = []
    #     l_per_n = self._sample_dist

    #     if len(self._waypoints) == 0:
    #       l = np.linalg.norm(np.array(self._xend[:3])-np.array(self._xinit[:3]))
    #       Ns.append(int(l/l_per_n))
    #       return Ns

    #     # Samples to reach the first gate
    #     l = np.linalg.norm(np.array(self._waypoints[0])-np.array(self._xinit[:3]))
    #     Ns.append(int(l/l_per_n))
    #     # Samples for each trajectory segment
    #     for i in range(self._wpt_num - 1):
    #         l = np.linalg.norm(np.array(self._waypoints[i])-np.array(self._waypoints[i+1]))
    #         Ns.append(int(l/l_per_n))
    #     # The last gate's position is the terminal position
    #     l = np.linalg.norm(self._xend[:3] - np.array(self._waypoints[-1]))
    #     Ns.append(int(l/l_per_n)) 
    #     return Ns
               
    def print(self):
        print("wpt_num: ", self._wpt_num)
        print("tol_wpt: ", self._tol_wpt)
        print("tol_term: ", self._tol_term)
        print("waypoints: ", self._waypoints)

  


