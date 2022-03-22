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
            self.zref=2.3
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

    def FluxP3D_McD2003_kpkp_hMpc(self, z, k_perp, k_par, linear=False, k_star = np.inf):
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
    
    def meanFlux(self, z):
        tau_eff = 2.53e-3 * (1+z)**3.7
        mean_F = np.exp(-tau_eff)
        return mean_F
    
    def PLE_Phi(self, Mg, z, Mg_star, Phi_star, alpha=-4.31, beta=-1.54):
        """
        Eq 6 in https://arxiv.org/pdf/1509.05607 with default values from Table 4
        """

        Phi = Phi_star / (10**(0.4*(alpha + 1)*(Mg - Mg_star)) + 10**(0.4*(beta + 1)*(Mg - Mg_star)))

        return Phi
    
    def PLE_Mg_star(self, z, Mg_star_zp, zp=2.2, k1=0.08, k2=-0.40):
        """
        Eq 7 in https://arxiv.org/pdf/1509.05607 with default values from Table 4
        """
        Mg_star = Mg_star_zp - 2.5*(k1*(z - zp) + k2*(z - zp)**2)

        return Mg_star

    
    def LEDE_Phi_star(self, z, Phi_star_zp, zp=2.2, c1a=-.14, c1b=0.32):
        """
        Eq 8 in https://arxiv.org/pdf/1509.05607 with default values from Table 4
        Note: We exponentiate here instead of leaving log(Phi_star) as in the paper
        """
        log_Phi_star = np.log(Phi_star_zp) + c1a*(z - zp) + c1b*(z - zp)**2
        
        return np.exp(log_Phi_star)
    
    def LEDE_Mg_star(self, z, Mg_star_zp, c2, zp=2.2):
        """
        Eq 9 in https://arxiv.org/pdf/1509.05607 with default values from Table 4
        """
        return Mg_star_zp + c2 * (z - zp)
    
    def alpha(self, z, alpha_zp, c3=0.32, zp=2.2):
        """
        Eq 10 in https://arxiv.org/pdf/1509.05607 with default values from Table 4
        """
        return alpha_zp + c3 * (z - zp)
    
    def PLE_Only_Phi(self, Mg, z):
        #for z = 0.68-4.0
        Mg_star_zp = -26.71
        log_Phi_star = -6.01
        zp = 2.2
        
        if z >= 0.68 and z <= 2.2:
            alpha = -4.31
            beta = -1.54
            k1 = -0.08
            k2 = -0.40
            
            Mg_star = self.PLE_Mg_star(z, Mg_star_zp=Mg_star_zp, k1=k1, k2=k2)
            return self.PLE_Phi(Mg, z, Mg_star=Mg_star, Phi_star=10**(log_Phi_star), alpha=alpha, beta=beta)
        
        if z > 2.2 and z <= 4:
            alpha = -3.04
            beta = -1.38
            k1 = -0.25
            k2 = -0.05
            
            Mg_star = self.PLE_Mg_star(z, Mg_star_zp=Mg_star_zp, k1=k1, k2=k2)
            print(Mg_star)
            print(10**(log_Phi_star))
            return self.PLE_Phi(Mg, z, Mg_star=Mg_star, Phi_star=10**(log_Phi_star), alpha=alpha, beta=beta)
        
        
        
    def PLE_LEDE_Phi(self, Mg, z):
        Mg_star_zero = -22.25
        log_Phi_star_zero = -5.93
        zp = 2.2
        alpha = -3.89
        beta = -1.47
        
        if z >= 0.68 and z <= 2.2:
            k1 = 1.59
            k2 = -0.36
            
            Mg_star = self.PLE_Mg_star(z, Mg_star_zp=Mg_star_zero, k1=k1, k2=k2, zp=0)
            
            return self.PLE_Phi(Mg, z, Mg_star=Mg_star, Phi_star=10**(log_Phi_star_zero), alpha=alpha, beta=beta)
        
        if z > 2.2 and z <= 4:
            c1a = -0.46
            c1b = -0.06
            c2 = -0.14     
            c3 = 0.32

            alpha = self.alpha(z, alpha_zp=alpha, c3=c3, zp=2.2)

            Phi_star = self.LEDE_Phi_star(z, Phi_star_zp=10**(log_Phi_star_zero), zp=0, c1a=c1a, c1b=c1b)
            Mg_star = self.LEDE_Mg_star(z, Mg_star_zp=Mg_star_zero, c2=c2, zp=0)
            print((Mg_star))
            print(Phi_star)
            return self.PLE_Phi(Mg, z, Mg_star=Mg_star, Phi_star=Phi_star, alpha=alpha, beta=beta)
            
            
            
        
        
        
        
        
        
        
        
    