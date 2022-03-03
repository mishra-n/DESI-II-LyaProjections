import analytic_bias_McD2003 as bM03
import analytic_p1d_PD2013 as p1D
import theoryLya as tLya
import cosmoCAMB as cCAMB
import helper as helper

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

class Survey(object):
    """Survey object encodes specific survey strategy information as well physical distributions of objects and brightness
    nzr file: gives the number density of quasars per sq degree as a function of redshift and z """ #ADD QLF functionality later
    
    
    def __init__(self, nzr_file, survey_A, cosmo=None):
            x = np.genfromtxt(nzr_file, skip_header=7)
            self.nzr = x[:,2]
            self.zs = x[:,0]
            self.rs = x[:,1]
            self.z_bins = np.unique(self.zs)
            self.r_bins = np.unique(self.rs)
            self.dz = self.z_bins[1] - self.z_bins[0]
            self.dr = self.r_bins[1] - self.r_bins[0]
            
            self.surveyArea = survey_A
            if cosmo:
                self.cosmo=cosmo
                self.zref=cosmo.pk_zref
            else:
                self.zref=2.25
                self.cosmo=cCAMB.Cosmology(self.zref)            
            
    def quasar_comoving_space_density_temp(self, Mg, z):

        Phi_star = 10**(-6.01)
        Mg_star_zp = -26.71

        if z < 2.2:
            alpha = -4.31
            beta = -1.54
            k1 = -0.08
            k2 = -0.40

        if z >= 2.2: 
            alpha = -3.04
            beta = -1.38
            k1 = -0.25
            k2 = -0.05


        Mg_star = Mg_star_zp - 2.5*(k1*(z - 2.2) + k2*(z - 2.2)**2)

        Phi = Phi_star / (10**(0.4*(alpha + 1)*(Mg - Mg_star)) + 10**(0.4*(beta + 1)*(Mg - Mg_star)))

        return Phi           
            
            
