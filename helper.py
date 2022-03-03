import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interp
import os


def kpkp_to_kmu(k_perp, k_par):

    k = np.sqrt(k_par**2 + k_perp**2)
    mu = k_par / k

    return k, mu

def kmu_to_kpkp(k, mu):

    k_perp = k * np.sqrt((1 - mu**2))
    k_par = k * mu

    return k_perp, k_par