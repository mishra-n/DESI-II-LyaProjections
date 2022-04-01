import analytic_bias_McD2003 as bM03
import analytic_p1d_PD2013 as p1D
import theoryLya as tLya
import cosmoCAMB as cCAMB
import helper as helper
from astropy.table import Table
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp

class Survey(object):
    """Survey object encodes specific survey strategy information as well physical distributions of objects and brightness
    nzr file: gives the number density of quasars per sq degree as a function of redshift and z """ #ADD QLF functionality later
    
    
    def __init__(self, from_file=1, snr_list='./lya-snr-guadalupe.fits', nzr_file='./nzs/nzr_qso.dat', k_corr='mcgreer2013', survey_A=16000, z_range=None, r_range=None, dz=0.2):
        
        self.low_lambda = 1041
        self.high_lambda = 1185
        if from_file==1:
            """
            All stuff the utilize the z and r bins provided by the files. This can all be reworked to fit your needs. In the end you need and n(z,r) where the first dimension lists the zs and the second dimension lists the rs, and the value at the coordinate is the number density within the bin.
            """
            x = np.genfromtxt(nzr_file, skip_header=3)
            self.nzr_temp = x[:,2]
            self.zs = x[:,0]
            self.rs = x[:,1]
            if z_range is None:
                self.z_bins = np.unique(self.zs)
            else:
                self.z_bins = np.unique(self.zs)[np.logical_and(self.zs <max(z_range), self.zs > min(z_range))]
                
            if r_range is None:
                self.r_bins = np.unique(self.rs)
            else:
                self.r_bins = np.unique(self.rs)[np.logical_and(self.rs <max(r_range), self.rs > min(z_range))]
                
            self.dz = self.z_bins[1] - self.z_bins[0]
            self.dr = self.r_bins[1] - self.r_bins[0]
            
            
            """All of this n(z,r) stuff is specific to my use case. In reality all you should need is an input in the shape below"""
            self.nzr = np.zeros(shape=(self.z_bins.shape[0], self.r_bins.shape[0]))
            
            for i, z_val in enumerate(self.z_bins):
                for j, r_val in enumerate(self.r_bins):
                    self.nzr[i,j] = self.nzr_temp[np.logical_and(self.zs==z_val, self.rs==r_val)]
            
            self.surveyArea = survey_A
            
            self.individualSNR = False


                    
        if from_file==2:
            self.SNRdata = Table.read(snr_list)
            self.zs = self.SNRdata['Z']
            self.rs = 22.25
            self.dz = 0.2
            self.dr = 0.5

            self.z_edges = np.arange(self.SNRdata['Z'].min().round(decimals=0), self.SNRdata['Z'].max().round(decimals=0), self.dz)
            
            self.z_bins = (self.z_edges[:-1] + self.z_edges[1:]) / 2
            
            self.r_bins = np.unique(self.rs)
            self.surveyArea = survey_A
            
            self.individualSNR = True
            
            n, bin_edges = np.histogram(self.zs, self.z_edges)
            
            self.nzr = n[:,None] / self.surveyArea / self.dz / self.dr
            
        self.k_corr = self.get_k_corr(paper=k_corr)
    
    def get_k_corr(self, paper='mcgreer2013'):
        """
        Pick between McGreer (2013) k-correction or Croom (2009) k-correction.
        We use WebPlotDigitizer to extract from Figure 9 from  https://arxiv.org/pdf/1509.05607
        """
        paper_low = paper.lower()
        k_correction = np.zeros
        
        if (paper_low == "mcgreer2013") or (paper_low == 'mcgreer'):
            k_correction = np.loadtxt('k-correction_McGreer2013.txt')
            
            z = k_correction[:,0]
            k_corr = k_correction[:,1]

            k_corr_func = interp.interp1d(z, k_corr, bounds_error=False)

            return k_corr_func
            
        if (paper_low == "croom2009") or (paper_low == 'croom'):
            k_correction = np.loadtxt('k-correction_Croom2009.txt')
        
            z = k_correction[:,0]
            k_corr = k_correction[:,1]

            k_corr_func = interp.interp1d(z, k_corr, bounds_error=False)

            return k_corr_func
        
    def g_mag_from_abs(self, Mg, z):
        """
        Equation 4 in https://arxiv.org/pdf/1509.05607
        """
        return Mg + dm(z) + (self.k_corr(z) - self.k_corr(z=2))