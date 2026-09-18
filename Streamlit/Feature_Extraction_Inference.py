import pywt
import numpy as np

def extract_features(Ia,Ib,Ic) :
    Ia = np.asarray(Ia,dtype = float).flatten()
    Ib = np.asarray(Ib,dtype = float).flatten()
    Ic = np.asarray(Ic,dtype = float).flatten()
    if len(Ia) != 80 :
        raise ValueError('Ia must contain exactly 80 samples.')
    if len(Ib) != 80 :
        raise ValueError('Ib must contain exactly 80 samples.')
    if len(Ic) != 80 :
        raise ValueError('Ic must contain exactly 80 samples.')

    rms_a = np.sqrt(np.mean(Ia ** 2))
    rms_b = np.sqrt(np.mean(Ib ** 2))
    rms_c = np.sqrt(np.mean(Ic ** 2))

    peak_a = np.max(np.abs(Ia))
    peak_b = np.max(np.abs(Ib))
    peak_c = np.max(np.abs(Ic))

    std_a = np.std(Ia,ddof = 1)
    std_b = np.std(Ib,ddof = 1)
    std_c = np.std(Ic,ddof = 1)

    crest_a = peak_a / (rms_a + np.finfo(float).eps)
    crest_b = peak_b / (rms_b + np.finfo(float).eps)
    crest_c = peak_c / (rms_c + np.finfo(float).eps)

    Ia_fft = np.fft.fft(Ia)
    Ib_fft = np.fft.fft(Ib)
    Ic_fft = np.fft.fft(Ic)
    thd_a = (np.linalg.norm(np.abs(Ia_fft[2:])) / (np.abs(Ia_fft[1]) + np.finfo(float).eps))
    thd_b = (np.linalg.norm(np.abs(Ib_fft[2:])) / (np.abs(Ib_fft[1]) + np.finfo(float).eps))
    thd_c = (np.linalg.norm(np.abs(Ic_fft[2:])) / (np.abs(Ic_fft[1]) + np.finfo(float).eps))

    I_A = Ia_fft[1]
    I_B = Ib_fft[1]
    I_C = Ic_fft[1]
    a = np.exp(1j * 2 * np.pi / 3)
    I0 = np.abs((I_A + I_B + I_C) / 3)
    I1 = np.abs((I_A + a * I_B + (a ** 2) * I_C) / 3)
    I2 = np.abs((I_A + (a ** 2) * I_B + a * I_C) / 3)
    unbalance = I2 / (I1 + np.finfo(float).eps)

    level = 5
    wavelet_name = 'db4'
    coeffs_a = pywt.wavedec(Ia,wavelet_name,level = level)
    coeffs_b = pywt.wavedec(Ib,wavelet_name,level = level)
    coeffs_c = pywt.wavedec(Ic,wavelet_name,level = level)

    D5_a = coeffs_a[1]
    D4_a = coeffs_a[2]
    D3_a = coeffs_a[3]
    D2_a = coeffs_a[4]
    D1_a = coeffs_a[5]

    D5_b = coeffs_b[1]
    D4_b = coeffs_b[2]
    D3_b = coeffs_b[3]
    D2_b = coeffs_b[4]
    D1_b = coeffs_b[5]

    D5_c = coeffs_c[1]
    D4_c = coeffs_c[2]
    D3_c = coeffs_c[3]
    D2_c = coeffs_c[4]
    D1_c = coeffs_c[5]

    energy_a_D1 = np.sum(D1_a ** 2)
    energy_a_D2 = np.sum(D2_a ** 2)
    energy_a_D3 = np.sum(D3_a ** 2)
    energy_a_D4 = np.sum(D4_a ** 2)
    energy_a_D5 = np.sum(D5_a ** 2)

    energy_b_D1 = np.sum(D1_b ** 2)
    energy_b_D2 = np.sum(D2_b ** 2)
    energy_b_D3 = np.sum(D3_b ** 2)
    energy_b_D4 = np.sum(D4_b ** 2)
    energy_b_D5 = np.sum(D5_b ** 2)

    energy_c_D1 = np.sum(D1_c ** 2)
    energy_c_D2 = np.sum(D2_c ** 2)
    energy_c_D3 = np.sum(D3_c ** 2)
    energy_c_D4 = np.sum(D4_c ** 2)
    energy_c_D5 = np.sum(D5_c ** 2)

    def calculate_entropy(D) :
        energy = np.sum(D ** 2)
        p = (D ** 2) / (energy + np.finfo(float).eps)
        return -np.sum(p * np.log2(p + np.finfo(float).eps))

    entropy_D1_a = calculate_entropy(D1_a)
    entropy_D2_a = calculate_entropy(D2_a)
    entropy_D3_a = calculate_entropy(D3_a)
    entropy_D4_a = calculate_entropy(D4_a)
    entropy_D5_a = calculate_entropy(D5_a)

    var_D1_a = np.var(D1_a,ddof = 1)
    var_D2_a = np.var(D2_a,ddof = 1)
    var_D3_a = np.var(D3_a,ddof = 1)
    var_D4_a = np.var(D4_a,ddof = 1)
    var_D5_a = np.var(D5_a,ddof = 1)

    std_D1_a = np.std(D1_a,ddof = 1)
    std_D2_a = np.std(D2_a,ddof = 1)
    std_D3_a = np.std(D3_a,ddof = 1)
    std_D4_a = np.std(D4_a,ddof = 1)
    std_D5_a = np.std(D5_a,ddof = 1)

    features = np.array([rms_a,rms_b,rms_c,
                         peak_a,peak_b,peak_c,
                         std_a,std_b,std_c,
                         crest_a,crest_b,crest_c,
                         thd_a,thd_b,thd_c,
                         I0,I1,I2,unbalance,
                         energy_a_D1,energy_a_D2,energy_a_D3,energy_a_D4,energy_a_D5,
                         energy_b_D1,energy_b_D2,energy_b_D3,energy_b_D4,energy_b_D5,
                         energy_c_D1,energy_c_D2,energy_c_D3,energy_c_D4,energy_c_D5,
                         entropy_D1_a,entropy_D2_a,entropy_D3_a,entropy_D4_a,entropy_D5_a,
                         var_D1_a,var_D2_a,var_D3_a,var_D4_a,var_D5_a,
                         std_D1_a,std_D2_a,std_D3_a,std_D4_a,std_D5_a],dtype = float)

    if len(features) != 49 :
        raise ValueError(f'Feature extraction returned {len(features)} features instead of 49.')
    return features