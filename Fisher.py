import analytic_bias_McD2003 as bM03
import analytic_p1d_PD2013 as p1D
import theoryLya as tLya
import cosmoCAMB as cCAMB
import helper as helper
from astropy.table import Table
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp



class Fisher(object):
    def __init__(self, effectiveStatistics, k_bins, mu_bins):
        self.Survey = effectiveStatistics.Survey
        self.Spectrograph = effectiveStatistics.Spectrograph
        self.theoryLya = effectiveStatistics.theoryLya
        self.effectiveStatistics = effectiveStatistics
        
        self.ks = k_bins
        self.mus = mu_bins
        self.dlnk = np.log(self.ks[1]) - np.log(self.ks[0])
        self.dmu =  self.mus[1] - self.mus[0]
        
    def k_fid(k_perp, k_par, a_perp, a_par):
        return k_perp/a_perp, k_par/a_par
    
    def dPdtheta(self, z, k_hMpc, mu, linear=False, theta=None, b_delta=np.sqrt(0.0173), beta=1.58):
        P_fid = self.theoryLya.FluxP3D_McD2003_hMpc(z=z, k_hMpc=k_hMpc, mu=mu, linear=linear)
        if theta == 'alpha_parallel':
            return -P_fid - mu*(1-mu**2)*self.theoryLya.dP3Ddmu(z, k_hMpc, mu) - k_hMpc*mu**2*self.theoryLya.dP3Ddk(z, k_hMpc, mu)
        if theta == 'alpha_perp':
            return -2*P_fid + mu*(1-mu**2)*self.theoryLya.dP3Ddmu(z, k_hMpc, mu) - k_hMpc*(1-mu**2)*self.theoryLya.dP3Ddk(z, k_hMpc, mu)
        if theta == 'beta':
            return self.theoryLya.dP3Ddbeta(z, k_hMpc, mu, b_delta=b_delta, beta=beta)
        if theta == 'b_delta':
            return self.theoryLya.dP3Ddbdelta(z, k_hMpc, mu, b_delta=b_delta, beta=beta)
    
    def Cov_prefactor(self):
        return (4.*np.pi**2.) / (self.dlnk*self.dmu*self.effectiveStatistics.surveyVolume[None,:]*self.ks[:,None]**3.)
    
        
    def Cov_kmtz(self, k_los=0.1, r_max = 22.5, t_range = np.arange(1000,13000,1000), n_eff=None):
        prefactor = self.Cov_prefactor()
        Ptot = self.effectiveStatistics.Ptot_kmtz(self.ks, self.mus, k_los= k_los, r_max = r_max, t_range=t_range)

        return prefactor[:,None,None,:] * Ptot**2
    
    def derivatives_a_kmz(self, k_los=0.1, basis=np.array(['alpha_parallel', 'alpha_perp']), poly=True):
        
        N = len(basis)
        N += 15
        derivatives = np.zeros(shape=(N, self.ks.shape[0], self.mus.shape[0], self.Survey.z_bins.shape[0]))
        for n in range(len(basis)):
            print(n)
            for i,k_val in enumerate(self.ks):
                for j, mu_val in enumerate(self.mus):
                    derivatives[n, i, j, :] = self.dPdtheta(self.Survey.z_bins, k_val, mu_val, linear=False, theta=basis[n])

        if poly==True: 
            for n in range(len(basis), N):
                print(n)
                for i,k_val in enumerate(self.ks):
                    for j, mu_val in enumerate(self.mus):    
                        m = n - len(basis)
                        derivatives[n, i, j, :] = (mu_val**(2*(m//5)) * k_val**(m%5))[None]

        return derivatives
    
    def Fisher_ab_tz(self, k_los=0.1, r_max = 22.5, t_range = np.arange(1000,13000,1000), basis=np.array(['alpha_parallel', 'alpha_perp'])):
        
        Covariance_kmtz = self.Cov_kmtz(k_los, r_max, t_range)
        derivatives = self.derivatives_a_kmz(k_los, basis=basis, poly=True)
        
        N = len(basis)
        N += 15
        
        Fisher_ab_kmtz = np.zeros(shape=(N,N, self.ks.shape[0], self.mus.shape[0], t_range.shape[0], self.Survey.z_bins.shape[0]))
        for n in range(N):
            for m in range(N):
                Fisher_ab_kmtz[n,m,:,:,:,:] = derivatives[n,:,:,None,:] * (1/Covariance_kmtz) * derivatives[m,:,:,None,:]

        Fisher_ab_tz = Fisher_ab_kmtz.sum(axis=2).sum(axis=2) #summing over all km pairs
        
        return Fisher_ab_tz