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
    
    
    def __init__(self, from_file=True, nzr_file='./nzs/nzr_qso.dat', survey_A=16000, cosmo=None, z_bins=np.arange(2, 5, .25), r_bins=np.arange(19.25, 23.75, .5)):
        
        if from_file is True:
            x = np.genfromtxt(nzr_file, skip_header=3)
            self.nzr_temp = x[:,2]
            self.zs = x[:,0]
            self.rs = x[:,1]
            self.z_bins = np.unique(self.zs)
            self.r_bins = np.unique(self.rs)
            self.dz = self.z_bins[1] - self.z_bins[0]
            self.dr = self.r_bins[1] - self.r_bins[0]
            
            self.nzr = np.zeros(shape=(self.z_bins.shape[0], self.r_bins.shape[0]))
            for i, z_val in enumerate(self.z_bins):
                for j, r_val in enumerate(self.r_bins):
                    self.nzr[i,j] = self.nzr_temp[np.logical_and(self.zs==z_val, self.rs==r_val)]
            
            self.surveyArea = survey_A
            if cosmo:
                self.cosmo=cosmo
                self.zref=cosmo.pk_zref
            else:
                self.zref=2.25
                self.cosmo=cCAMB.Cosmology(self.zref)
        else:
            n_zr = np.zeros(shape=(rs.shape[0], zs.shape[0]))
            
            for i,z in enumerate(z_bins):
                n_zr[:,i] = np.sum(self.quasarComovingSpaceDensity(r_bins, z)) 
            
            
    def quasarComovingSpaceDensity(self, Mg, z):

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
    
    def n_rz():
        if z_bins==None and r_bins==None:
            
            n_rz = np.zeros(shape=(rs.shape[0], zs.shape[0]))

            Mg_space = np.linspace(Mg_min, Mg_max, dMgs)

            for i,z in enumerate(zs):
                n_z[i] = np.sum(self.quasarComovingSpaceDensity(Mg_space, z) * dMgs)

        return n_z
            
        
        
        
    
