import numpy as np
import scipy.interpolate
import camb
from camb import model, initialpower

class Cosmology(object):
    """Compute cosmological functions using CAMB.

       Plenty of room for improvement, for now just something to get started."""

    def __init__(self,pk_zref=None):
        """Setup cosmological model.

            If pk_zref is set, it will compute linear power at z=pk_zref."""
        self.pars = camb.CAMBparams()
        #self.pars.set_cosmology(H0=67.5,ombh2=0.022,omch2=0.122)
        #self.pars.InitPower.set_params(As=2e-9,ns=0.965, r=0)
        # use same parameters as in c++ code
        ob = 0.046
        om = 0.286
        oc = om-ob
        self.h = 0.719
        self.c = 3e5
        self.pars.set_cosmology(H0=100.0*self.h,ombh2=ob*(self.h**2),omch2=oc*(self.h**2))
        self.pars.InitPower.set_params(As=2.2e-9,ns=0.961, r=0)
        self.pk_zref=pk_zref
        if self.pk_zref:
            self.pars.set_matter_power(redshifts=[self.pk_zref], kmax=10.0)
            # compute and store linear power spectrum (at zref)
            self.pars.NonLinear = model.NonLinear_none
            self.results = camb.get_results(self.pars)
        else:
            # compute only background expansion
            self.results = camb.get_results(self.pars)
        # not sure where to put this
        self.c_kms = 2.998e5
        self.lya = 1215.67
        self.chi_z = lambda x: self.results.comoving_radial_distance(x) * self.h
        self.H_z = lambda x: self.results.hubble_parameter(x) / self.h
        self.E_z = lambda x: self.H_z(x) / self.H_z(0)
        

    def LinPk_hMpc(self,kmin=1.0e-4,kmax=1.0e1,npoints=1000):
        """Return linear power interpolator in units of h/Mpc, at zref"""
        if self.pk_zref:
            kh,_,pk = self.results.get_matter_power_spectrum(minkh=kmin,
                                                    maxkh=kmax,npoints=npoints)
            return scipy.interpolate.interp1d(kh,pk[0,:], bounds_error=False, fill_value=0)
        else:
            print('if you want LinPk_hMpc, initialize Cosmology with pk_zref')
            raise SystemExit

    def dkms_dhMpc(self,z):
        """Convertion factor from Mpc/h to km/s, at redshift z."""
        return self.results.hubble_parameter(z)/self.pars.H0/(1+z)*100.0

    def dkms_dlobs(self,z):
        """Convertion factor from lambda_obs to km/s, at redshift z."""
        return self.c_kms / self.lya_A / (1+z) 

    def dhMpc_dlobs(self,z):
        """Convertion factor from lambda_obs to Mpc/h, at redshift z."""
        return self.dkms_dlobs(z) / self.dkms_dhMpc(z)

    def dhMpc_ddeg(self,z):
        """Convertion factor from degrees to Mpc/h, at redshift z."""
        dMpc_drad=self.results.angular_diameter_distance(z)*(1+z)
        #print('dMpc_drad',dMpc_drad)
        return dMpc_drad*np.pi/180.0*self.pars.H0/100.0
    
    def V(self, z, low_lambda, max_lambda, area, dz):
        """
        Takes in redshift, bin width and survey area and returns volume element of that ring Mpc^3 h^-3
        """
        return self.L(z, low_lambda, max_lambda) * self.A(z, area) * dz
    
    def chi_trans(self,z):
        
        return self.chi_z(z) / (1+z)
    
    
    def L(self,z, low_lambda, max_lambda):
        return self.chi_z(z) - self.chi_z(low_lambda/max_lambda *(z + 1) - 1)
        
    def A(self, z, area):
        """
        Takes in redshift and area and returns area in comoving Mpc^2 h^-2SS
        z: redshift plane
        surveyArea: survey area in deg^2
        """
        return (self.chi_trans(z) * np.sqrt(area) * np.pi / 180)**2
        
    def dVdivdz(self, z, area):
        return ((1+z)**2 * 3 * self.chi_z(z)**2 * self.c_kms / self.H_z(z) - self.chi_z(z)**3 * 2 * (1+z))/(1+z)**4

