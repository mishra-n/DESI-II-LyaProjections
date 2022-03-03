import theoryLya as tLya
import cosmoCAMB as cCAMB
import helper as helper
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

redshifts = np.arange(1, 5.25, .25)
chimps = np.array([4754, 5569, 6275, 6901, 7450, 7939, 8378, 8775, 9132, 9463, 9765, 10043, 10299, 10540, 10761, 10972, 11165])
angular_diameter = interp.interp1d(redshifts, chimps)

class effectiveStatistics(object):
    
    def __init__(self, theoryLya, Survey, Spectrograph):
        self.theoryLya = theoryLya
        self.Survey = Survey
        self.Spectrograph = Spectrograph
       
    def v_n(self, t, r, z, k):
        P = self.theoryLya.FluxP1D_PD2013_hMpc(z, k)
        P_Nn = self.Spectrograph.P_Nn(t, r, z)
        
        return P / (P_Nn + P)
    
    def n_eff_trz(self, k, t_range = np.arange(1000,13000,1000), r_range = None, z_range = None):
        if r_range == None:
            r_range = self.Survey.r_bins
        if z_range == None:
            z_range = self.Survey.z_bins
        
        n_eff_trz = np.zeros(shape=(t_range.shape[0], r_bins.shape[0], z_range.shape[0]))
        
        for i, t_val in enumerate(t_range):
            for j, r_val in enumerate(r_range):
                for k, z_val in enumerate(z_range):
                    v_n_value * self.v_n(t_val, r_val, z_val, k) * self.Survey.dz * self.Survey.dr
                    
                    term = v_n_val / (angular_diameter(z_val))**2
                    
                    n_eff_trz[i,j,k] = term
                    
        return n_eff_trz, t_range, r_range, z_range