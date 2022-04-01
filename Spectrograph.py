import helper
import numpy as np
import scipy.interpolate as interp
import theoryLya as tLya


class Spectrograph(object):
    """
    Spectrograph statistics
    """
    def __init__(self, theoryLya, name='desi'):
        #function for spectrograph snr as a function of integration time, r-magnitude of quasar, redshift of quasar
        self.theoryLya = theoryLya
        self.name = name
        if self.name is 'desi':
            path = "/global/homes/n/nishant/DESI2_LyaProjections/DESI-II-lyaforecast/build/lib/desi_SNR/"
            self.snr = self.SNR(path)
            self.resolution = 1 #angstroms
        if self.name is 'weave':
            self.snr = self.SNR()
            self.resolution = 2.6 #angstroms (Kraljic et al (2022), 2201.02606)
            
            
        


    def openSNR(self, data):
        """
        returns SNR in a grid along wavelengths and redshifts for a given exposure time and magnitude
        """
        x = np.genfromtxt(data, skip_header=11)
        wavelengths = x[:,0]
        redshifts = np.arange(2, 5, .25)

        return x[:,1::], wavelengths, redshifts

    def meanSNR(self, SNR, wavelengths, redshifts):
        mean_SNR = np.zeros(shape=redshifts.shape[0])
        for i,z in enumerate(redshifts):
            selected_wavelengths = np.logical_and(1215*z < (wavelengths - 1215), 912*z < (wavelengths - 912))
            mean_SNR[i] = SNR[selected_wavelengths].mean()

        return mean_SNR, redshifts

    def SNR(self, path=None):
        if self.name is 'desi':
            ts = np.arange(1000,13000,1000)
            rs = np.arange(19.25, 23.75, .5)
            zs = np.arange(2, 5, .25)

            SNR_arr = []
            for j, r in enumerate(rs):
                for i, t in enumerate(ts):
                    file_name = 'toto-r' + str(rs[j]) + '-t' + str(ts[i]) + '-nexp4.dat'
                    SNR_val, wavelengths, redshifts = self.openSNR(path + file_name)
                    mean_SNR, redshifts = self.meanSNR(SNR_val, wavelengths, redshifts)
                    SNR_arr = np.append(SNR_arr, mean_SNR)

            Ts, Rs, Zs = np.meshgrid(ts, rs, zs)
            SNR_func = interp.LinearNDInterpolator(list(zip(Ts.flatten(), Rs.flatten(), Zs.flatten())), SNR_arr.flatten())
        
        if self.name is 'weave':
            file_name = 'weave_SNR.txt'
            x = np.loadtxt(file_name)
            snr = x[:,1]
            r = x[:,0]
            f_snr_r = interp.interp1d(r, snr)
            
            SNR_func = lambda ts, rs, zs : f_snr_r(rs) * np.sqrt(ts)
                    
        return SNR_func
    
    def P_Nn(self, t, r, z, SNR=None):
        """
        Eq 3 in McQuinn, White (2011): Noise power calculated from SNR per pixel as function of redshift and mean flux
        Note: k, mu independent Pn means that the derivatives evaluate to zero.
        """
        
        if SNR is None:
            P_Nn = self.snr(t, r, z)**(-2) * self.resolution / self.theoryLya.cosmo.lya * self.theoryLya.cosmo.H_z(z) #* tLya.meanFlux(z)**(-2)
        else:
            P_Nn = SNR**(-2) * self.resolution / self.theoryLya.cosmo.lya * self.theoryLya.cosmo.H_z(z)

        return P_Nn