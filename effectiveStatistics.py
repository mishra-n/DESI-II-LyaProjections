import theoryLya as tLya
import cosmoCAMB as cCAMB
import helper as helper
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

class effectiveStatistics(object):
    
    def __init__(self, theoryLya, Survey, Spectrograph):
        self.theoryLya = theoryLya
        self.Survey = Survey
        self.Spectrograph = Spectrograph
        
        self.surveyVolume = self.theoryLya.cosmo.V(self.Survey.z_bins, self.Survey.low_lambda, self.Survey.high_lambda, self.Survey.surveyArea, self.Survey.dz)
       
    def v_n(self, t, r, z, k, SNR=None):
        P = self.theoryLya.FluxP1D_PD2013_hMpc(z, k)
        if SNR is None:
            P_Nn = self.Spectrograph.P_Nn(t, r, z)
        else:
            P_Nn = self.Spectrograph.P_Nn(t, r, z, SNR=SNR)
        
        return P / (P_Nn + P)
    
        
    def n_eff_trz(self, k_val, t_range = np.arange(1000,13000,1000)):
        r_range = self.Survey.r_bins
        z_range = self.Survey.z_bins

        n_eff_trz = np.zeros(shape=(t_range.shape[0], r_range.shape[0], z_range.shape[0]))
        
        for i, t_val in enumerate(t_range):
            for k, z_val in enumerate(z_range):
                for j, r_val in enumerate(r_range):
                    nzr_val = self.Survey.nzr[k,j]
                    #print(nzr_val)
                    if self.Survey.individualSNR is False:
                        v_n_val = self.v_n(t_val, r_val, z_val, k_val) * nzr_val * self.Survey.surveyArea* self.Survey.dz * self.Survey.dr
                        area = self.theoryLya.cosmo.A(z_val, self.Survey.surveyArea)
                        term = v_n_val / area
                        n_eff_trz[i,j,k] = term
                    
                    if self.Survey.individualSNR is True:
                        qso=self.Survey.SNRdata[(self.Survey.SNRdata['Z']>z_val)&(self.Survey.SNRdata['Z']<(z_val + self.Survey.dz))]
                        v_n_val = self.v_n(t_val, r_val, qso['Z'], k_val, SNR = qso['SNR'] * np.sqrt(t_val/4000)).sum()
                        area = self.theoryLya.cosmo.A(z_val, self.Survey.surveyArea)

                        term = v_n_val / area
                        n_eff_trz[i,j,k] = term

        return n_eff_trz
    
    def V_eff_kmtz(self, k, mu, k_los, n_eff=None, r_max = 22.5):
        Plos = self.theoryLya.FluxP1D_PD2013_hMpc(self.Survey.z_bins, k_los)
        
        if n_eff==None:
            n_eff = np.nan_to_num(self.n_eff_trz(k_los, t_range=np.arange(1000,13000,1000)))
            
        n_eff_tz = n_eff[:, self.Survey.r_bins < r_max, :].sum(axis=1)

        V_eff = np.zeros(shape=(len(k), len(mu), n_eff_tz.shape[0], n_eff_tz.shape[1]))
        
        for i,k_val in enumerate(k):
            for j, mu_val in enumerate(mu):
                Pk = self.theoryLya.FluxP3D_McD2003_hMpc(z=self.Survey.z_bins, k_hMpc=k_val, mu=mu_val)
                
                V_eff[i,j,:,:] = (Pk[None,:] / (Pk[None,:] + Plos[None,:] / n_eff_tz))**2 * self.surveyVolume[None,:]
                
        return V_eff
    
    def Ptot_kmtz(self, k, mu, k_los, n_eff=None, r_max = 22.5, t_range = np.arange(1000,13000,1000)):
        if n_eff==None:
            n_eff = np.nan_to_num(self.n_eff_trz(k_los, t_range=t_range))
        
        n_eff = n_eff[:, self.Survey.r_bins < r_max, :].sum(axis=1)
                
        Plos = self.theoryLya.FluxP1D_PD2013_hMpc(self.Survey.z_bins, k_los)

        P_tot_kmtz = np.zeros(shape=(k.shape[0], mu.shape[0],t_range.shape[0], self.Survey.z_bins.shape[0]))
        
        for i,k_val in enumerate(k):
            for j,mu_val in enumerate(mu):
                Pk = self.theoryLya.FluxP3D_McD2003_hMpc(z=self.Survey.z_bins, k_hMpc=k_val, mu=mu_val)
                P_tot_kmtz[i,j,:,:] = Pk[None,:] + Plos[None,:]/n_eff
        return P_tot_kmtz
        
        
        
        
        
        
        