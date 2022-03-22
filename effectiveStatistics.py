import theoryLya as tLya
import cosmoCAMB as cCAMB
import helper as helper
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

#redshifts = np.array([.2, .35, .5, .75, 1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4, 4.25,4.5,4.75,5])
#chimps = np.array([1185.16, 1986.9, 2731.93, 3827.05, 4754, 5569, 6275, 6901, 7450, 7939, 8378, 8775, 9132, 9463, 9765, 10043, 10299, 10540, 10761, 10972, 11165])
#angular_diameter = interp.interp1d(redshifts, chimps)

class effectiveStatistics(object):
    
    def __init__(self, theoryLya, Survey, Spectrograph):
        self.theoryLya = theoryLya
        self.Survey = Survey
        self.Spectrograph = Spectrograph
       
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
    
    def V_eff_kmtrz(self, k, mu, n_eff_trz, k_los):
        Plos = self.theoryLya.FluxP1D_PD2013_hMpc( self.Survey.z_bins, k_los)
        V_eff = np.zeros(shape=(len(k), len(mu), n_eff_trz.shape[0], n_eff_trz.shape[1], n_eff_trz.shape[2]))
        
        for i,k_val in enumerate(k):
            for j, mu_val in enumerate(mu):
                Pk = self.theoryLya.FluxP3D_McD2003_hMpc(z=self.Survey.z_bins, k_hMpc=k_val, mu=mu_val)
                volume = self.theoryLya.cosmo.V(self.Survey.z_bins, self.Survey.low_lambda, self.Survey.high_lambda, self.Survey.surveyArea, self.Survey.dz)
                V_eff[i,j,:,:,:] = (Pk[None,None,:] / (Pk[None,None,:] + Plos[None,None,:] / n_eff_trz))**2 * volume[None,None,:]
                
        return V_eff
        
        
        