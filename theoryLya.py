import analytic_bias_McD2003 as bM03
import analytic_p1d_PD2013 as p1D
import cosmoCAMB as cCAMB
import helper as helper

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

def meanFlux(z):
    """
    Eq 2.1 in Walther et al. (2012) for analytic mean flux
    """
    tau_eff = 2.53e-3 * (1+z)**3.7
    F = np.exp(-tau_eff)

    return F
    
class theoryLya(object):
    """Make predictions for Lyman alpha 3D P(z,k,mu).
        Should only be used at the level of Fisher forecasts.
        Uses CAMB to generate linear power, and McDonald (2003) for Lya stuff.
        All units internally are in h/Mpc."""

    def __init__(self, cosmo=None):
        """cosmo is an optional cosmoCAMB.Cosmology object"""
        if cosmo:
            self.cosmo=cosmo
            self.zref=cosmo.pk_zref
        else:
            self.zref=2.25
            self.cosmo=cCAMB.Cosmology(self.zref)

        # get linear power spectrum 
        self.kmin=1.0e-4
        self.kmax=1.0e2
        self.linPk = self.cosmo.LinPk_hMpc(self.kmin,self.kmax,1000)

    def FluxP3D_McD2003_hMpc(self,z,k_hMpc,mu,linear=False):
        """3D power spectrum P_F(z,k,mu). 

            If linear=True, it will ignore small scale correction."""
        # get linear power at zref
        k = np.fmax(k_hMpc,self.kmin)
        k = np.fmin(k,self.kmax)
        P = self.linPk(k)
        # get flux scale-dependent biasing (or only linear term)
        b = bM03.bias_hMpc_McD2003(k,mu,linear)
        # get (approximated) redshift evolution
        zevol = pow( (1+z)/(1+self.zref), 3.8)
        return P * b * zevol

    def FluxP1D_PD2013_hMpc(self,z,k_hMpc,res_hMpc=None,pix_hMpc=None):
        """Analytical P1D, in units of h/Mpc instead of km/s."""
        # transform to km/s
        dkms_dhMpc = self.cosmo.dkms_dhMpc(z)
        k_kms = k_hMpc / dkms_dhMpc
        # get analytical P1D from Palanque-Delabrouille (2013)
        P_kms = p1D.P1D_z_kms_PD2013(z,k_kms)
        P_hMpc = P_kms / dkms_dhMpc
        if res_hMpc:
            # smooth with Gaussian
            P_hMpc *= np.exp(-pow(k_hMpc*res_hMpc,2))
        if pix_hMpc:
            # smooth with Top Hat
            kpix = np.fmax(k_hMpc*pix_hMpc,1.e-5)
            P_hMpc *= pow(np.sin(kpix/2)/(kpix/2),2)
        return P_hMpc

    def FluxP3D_McD2003_kpkp_hMpc(self, z, k_perp, k_par, linear=True, k_star = np.inf):
        """
        3D power spectrum P_F(z, k_perp, kpar)
        
        Same as FluxP3D_hMpc but in k-space rather than k, mu space
        """

        k, mu  = helper.kpkp_to_kmu(k_perp, k_par)
    
        return self.FluxP3D_McD2003_hMpc(z, k , mu, linear=linear) * np.exp(-(k/k_star)**2)
    
    
    def FluxP1D_McD2003_hMpc(self, z, k_hMpc, k_star= np.inf):
        """
        Integrated P_F(z, k_perp, kpar) d^2 k_perp
        """
        
        p1d = np.zeros(shape=k_hMpc.shape[0])
        
        for i, k_par in enumerate(k_hMpc):
            integrand = lambda k_perp: self.FluxP3D_McD2003_kpkp_hMpc(z, k_perp, k_par, k_star=k_star) * 2 * np.pi * k_perp / (2*np.pi)**2
            
            p1d[i] = integrate.quad(integrand, 0, np.inf)[0]
        return p1d
    