import helper
import numpy as np
import scipy.interpolate as interp
import theoryLya as tLya


class Spectrograph(object):
    """
    Spectrograph statistics
    """
    def __init__(self, snr_path):
        #function for spectrograph snr as a function of integration time, r-magnitude of quasar, redshift of quasar
        self.snr = self.SNR(snr_path)


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

    def SNR(self, path):
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

        return SNR_func
    
    def P_Nn(self, t, r, z, SNR=None):
        """
        Eq 3 in McQuinn, White (2011): Noise power calculated from SNR per pixel as function of redshift and mean flux
        """

        delta_lambda = 1
        
        if SNR is None:
            P_Nn = 0.8 * tLya.meanFlux(z)**(-2) * self.snr(t, r, z)**(-2) * delta_lambda * ((1+z)/4)**(-3/2)
        else:
            P_Nn = 0.8 * tLya.meanFlux(z)**(-2) * SNR**(-2) * delta_lambda * ((1+z)/4)**(-3/2)

        return P_Nn