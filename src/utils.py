import numpy as np
import lmfit
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter
from warnings import warn
from math import fmod

def find_nearest(array: np.ndarray, value: float) -> tuple:
    """
    Finds the nearest value and its index in an array to a given value.

    Args:
        array (np.ndarray): The array to search.
        value (float): The value to find the nearest match for.

    Returns:
        tuple: A tuple containing the nearest value and its index in the array.
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def find_circle(x: np.ndarray, y: np.ndarray) -> tuple:
    """Finds a circle that best fits the given x, y data using Least Squares Circle Fit.
    
    Args:
        x (np.ndarray): Array of x positions of data in the complex plane (real).
        y (np.ndarray): Array of y positions of data in the complex plane (imaginary).
        
    Returns:
        tuple: Coordinates of the circle's center (x, y) and its radius (R).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    x, y = np.asarray(x), np.asarray(y)
    x_mean, y_mean = np.mean(x), np.mean(y)

    # Check if there are at least three unique points
    if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
        raise ValueError("At least three unique points are required to define a circle.")

    # Check for collinearity by calculating the area of the triangle formed by the first three unique points
    if len(x) >= 3:
        area = 0.5 * abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
        if area == 0:
            raise ValueError("The points are collinear and cannot define a circle.")

    # Center the data
    u, v = x - x_mean, y - y_mean

    # Matrix form of the system
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svvv + Suuv)])
    uc, vc = np.linalg.solve(A, B)

    # Radius and center
    center_x = uc + x_mean
    center_y = vc + y_mean
    radius = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(x))

    return center_x, center_y, radius

def find_circle2(x: np.array, y: np.array):
    '''
    implements the algebraic circle fitting technique detailed in Chernov & Lesort, and Probst, originally developed by Pratt
    '''
    #There is a bug in here somewhere -- the radius is not right
    z = x**2+y**2
    Mzz = np.sum(z**2)
    Mxz = np.sum(x*z)
    Myz = np.sum(y*z)
    Mxy = np.sum(x*y)
    Mxx = np.sum(x*x)
    Myy = np.sum(y*y)
    Mx = np.sum(x)
    My = np.sum(y)
    Mz = np.sum(z)
    n = len(x)

    #define the moments matrix
    M = np.array([[Mzz, Mxz, Myz, Mz],[Mxz, Mxx, Mxy, Mx],[Myz, Mxy, Myy, My],[Mz, Mx, My, n]])
    #This encodes the constraint B^2+C^2-4AD=1
    B = np.array([[0,0,0,-2],[0,1,0,0],[0,0,1,0],[-2,0,0,0]])

    w, vr = eig(M, b=B)
    #the eigenvalue w[i] corresponds to the eigenvector vr[:,i]

    #Find index of smallest positive eigenvalue
    ind = -1
    eta = float('inf')
    for i in range(len(w)):
        if w[i] > 0 and w[i] < eta:
            ind = i
            eta = w[i]

    if ind == -1:
        ValueError('Cannot find eigenvector in the circle fit')
    else:
        A = vr[0,ind]
        B = vr[1, ind]
        C = vr[2, ind]
        D = vr[3, ind]

    #normalize
    a = B**2+C**2-4*A*D
    A = A/a
    B = B/a
    C = C/a
    D = D/a

    xc = -B/(2*A)
    yc = -C/(2*A)
    R = 1/(2*np.abs(A))
    return xc, yc, R


def phase_centered(f, fr, Ql, theta, delay=0.):
    """
    Yields the phase response of a strongly overcoupled resonator
    in reflection, centered around the origin, with a potential linear
    background phase slope due to signal delay.

    Args:
        f: Frequency array.
        fr: Resonance frequency.
        Ql: Loaded quality factor (Ql ≈ Qc for Qi >> Qc).
        theta: Offset phase.
        delay (optional): Time delay between output and input signal,
                          leading to a linearly frequency-dependent phase shift.

    Returns:
        np.ndarray: Phase response for each frequency in `f`.
    """
    return theta - 2 * np.pi * delay * (f - fr) + 2. * np.arctan(2. * Ql * (1. - f / fr))

def phase_dist(angle):
    """
    Maps an angle from [-2pi, +2pi] to a phase distance on a circle [0, pi].
    
    Args:
        angle (np.ndarray or float): Angle(s) to be mapped, in radians.
        
    Returns:
        np.ndarray or float: The phase distance(s) corresponding to the input angle(s),
                             guaranteed to be in the range [0, pi].
    """
    return np.pi - np.abs(np.pi - np.abs(angle))

def periodic_boundary(angle):
    """
    Maps an arbitrary angle to the interval [-np.pi, np.pi).
    
    Args:
        angle (float or np.ndarray): Angle(s) in radians to be mapped.
    
    Returns:
        float or np.ndarray: The mapped angle(s) within [-np.pi, np.pi).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

#this needs to be changed to transform the circle to an arbitrary off-resonant point
def normalize(f_data: np.ndarray, z_data: np.ndarray, delay: float, a: float, alpha: float) -> np.ndarray:
    """
    Normalizes scattering data to a canonical position with the off-resonant
    point at (1, 0), adjusting for amplitude and phase offset, but not for
    rotation around the off-resonant point.

    Args:
        f_data (np.ndarray): The frequency data array, not used in the current implementation
                             but included for future extensions or adjustments.
        z_data (np.ndarray): The complex scattering data to be normalized.
        delay (float): The cable delay; currently not used in normalization but
                       included for consistency with calibration parameters.
        a (float): The normalization amplitude factor.
        alpha (float): The phase offset to be corrected.

    Returns:
        np.ndarray: The normalized scattering data.
    """
    z_norm = (z_data / a) * np.exp(-1j * alpha)
    return z_norm

def remove_delay(fdata: np.ndarray, sdata: np.ndarray, delay):
    return np.exp(2j*np.pi*delay*fdata)*sdata

def sloped_arctan(f, Ql, fr, delay, theta_0):
    offsetphase = theta_0 + -2*np.pi*f*delay + 2* np.arctan(2*Ql*(1-f/fr))
    return offsetphase

def partitionFrequencyBand(fdata: np.ndarray, GradS: np.ndarray, keep = 'above', cutoff = 0.5):
    '''
    Partitions the frequency band to separate data inside & outside the linewidth.

    Args:
        fdata: np.array of frequency data
        GradS: np.array derivative of the scattering data (dS/df)
        keep: string indicating whether to assign 1 or 0 to fdata with corresponding GradS above or below cutoff ratio
        cutoff: float between 0 and 1

    Returns:
        chiFunction: np.array of the same shape as fdata with entries 1 or 0 to indicate inside or outside linewidth
    '''

    #TODO: check if fdata and GradS have the same length
    GradSMagnitude = np.abs(GradS)
    chiFunction = np.zeros(len(fdata))

    cutoff_value = (1-cutoff)*np.min(GradSMagnitude) + cutoff*np.max(GradSMagnitude)
    if keep == 'above':
        for n in range(len(GradSMagnitude)):
            if GradSMagnitude[n] > cutoff_value:
                chiFunction[n] = 1  # set to one if |dS/df| is above the cutoff at this point
    elif keep == 'below':
        for n in range(len(GradSMagnitude)):
            if GradSMagnitude[n] < cutoff_value:
                chiFunction[n] = 1  # set to one if |dS/df| is below the cutoff at this point

    return chiFunction

def std_err_to_95pct(p: lmfit.Parameters):
    '''
    convert standard errors to 95% confidence intervals
    '''
    for key in p:
        p[key].stderr = 1.96*p[key].stderr

    return p

def remove_linear_sigma(fdata, sdata, sigma):
    return sdata*np.exp(2*np.pi*sigma*fdata)

def preprocess_delay(fdata, sdata, display_fit = False):
    '''
    Standalone function to find and remove the electrical delay.

    The "Chi function" is 1 close to resonance and zero away from resonance
    '''
    #smooth data and take derivative to remove asymmetry
    filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
    gradS = np.gradient(filtered_data, fdata)
    gradSmagnitude = np.abs(gradS)
    #estimate center frequency
    fc_index = np.argmax(gradSmagnitude)
    fc = fdata[fc_index]
    #estimate linewidth -- count up the spans of adjacent frequency points that are close to resonance
    chiFunction = partitionFrequencyBand(fdata, gradS)
    linewidth = np.dot(chiFunction[:-1], np.diff(fdata))

    #redefine the chi function to remove points farther away from fc for fitting
    chiFunction = np.zeros(len(fdata))
    for n in range(len(chiFunction)):
        if np.abs(fdata[n] - fc) < 6 * linewidth:
            chiFunction[n] = 1

    freq_arrays = np.split(fdata, [fc_index])
    data_arrays = np.split(sdata, [fc_index])
    chi_arrays = np.split(chiFunction, [fc_index])

    # TODO: trim data and perform two separate fits, average values for delay_guess

    delay_fit = []
    lrg_result = [0, 0]
    trimmed_freq = [0, 0]
    for n in range(2):
        # trim the data
        trim = np.nonzero(chi_arrays[n])
        trimmed_freq[n] = np.delete(freq_arrays[n], trim)
        trimmed_data = np.delete(data_arrays[n], trim)

        # fit trimmed freq and data
        # TODO: check the r**2 of these fits and warn the user if they're below 0.9
        trimmed_phase = np.unwrap(np.angle(trimmed_data))
        lrg_result[n] = linregress(trimmed_freq[n], trimmed_phase)
        delay_guess = lrg_result[n].slope / (-2 * np.pi)
        delay_fit = np.append(delay_fit, delay_guess)
    avg_delay_guess = (delay_fit[0] + delay_fit[1]) / 2
    # warm user if r^2 <0.9
    for n in range(2):
        if lrg_result[n].rvalue ** 2 < 0.9:
            warn(f'low r-squared in delay fit: {lrg_result[n].rvalue ** 2}')

    # plot the fits as a check
    if display_fit == True:
        plt.plot(fdata, np.unwrap(np.angle(sdata)))
        plt.plot(trimmed_freq[0], lrg_result[0].intercept + lrg_result[0].slope * trimmed_freq[0], color='k',
                 linestyle='dashed')
        plt.plot(trimmed_freq[1], lrg_result[1].intercept + lrg_result[1].slope * trimmed_freq[1], color='k',
                 linestyle='dashed')
        # plt.axvline(fc+6*linewidth)
        # plt.axvline(fc-6*linewidth)
        plt.ylabel('Phase (rad.)')
        plt.xlabel('Frequency')
        plt.show()

    #remove the delay and return processed data
    new_sdata = remove_delay(fdata, sdata, avg_delay_guess)
    return new_sdata

def preprocess_linear(fdata, sdata, display_fit = False):
    '''
            Just fit the magnitude (dB) while discarding data in the linewidth.

            Supposing S ~ exp(-2 pi sigma f), we are trying to extract sigma

            Args:
                fdata: numpy array of the frequency data
                sdata: numpy array of the scattering data

            Returns:
                delay: float representing the sigma in the model above
            '''
    filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
    gradS = np.gradient(filtered_data, fdata)
    gradSmagnitude = np.abs(gradS)
    # estimate center frequency
    fc_index = np.argmax(gradSmagnitude)
    fc = fdata[fc_index]
    # estimate linewidth -- count up the spans of adjacent frequency points that are close to resonance
    chiFunction = partitionFrequencyBand(fdata, gradS)
    linewidth = np.dot(chiFunction[:-1], np.diff(fdata))

    # redefine the chi function to remove points farther away from fc for fitting
    chiFunction = np.zeros(len(fdata))
    for n in range(len(chiFunction)):
        if np.abs(fdata[n] - fc) < 6 * linewidth:
            chiFunction[n] = 1

    freq_arrays = np.split(fdata, [fc_index])
    data_arrays = np.split(sdata, [fc_index])
    chi_arrays = np.split(chiFunction, [fc_index])

    # TODO: trim data and perform two separate fits, average values for delay_guess

    sigma_fit = []
    lrg_result = [0, 0]
    trimmed_freq = [0, 0]
    for n in range(2):
        # trim the data
        trim = np.nonzero(chi_arrays[n])
        trimmed_freq[n] = np.delete(freq_arrays[n], trim)
        trimmed_data = np.delete(data_arrays[n], trim)

        # fit trimmed freq and data
        trimmed_mag = 20 * np.log10(np.abs(trimmed_data))
        lrg_result[n] = linregress(trimmed_freq[n], trimmed_mag)
        sigma = -np.log(10) * lrg_result[n].slope / (40 * np.pi)
        sigma_fit = np.append(sigma_fit, sigma)
    avg_sigma = (sigma_fit[0] + sigma_fit[1]) / 2
    # warm user if r^2 <0.9
    for n in range(2):
        if lrg_result[n].rvalue ** 2 < 0.9:
            warn(f'low r-squared in linear background fit: {round(lrg_result[n].rvalue ** 2, 3)}')

    # plot the fits as a check
    if display_fit == True:
        plt.plot(fdata, 20 * np.log10(np.abs(sdata)))
        plt.plot(trimmed_freq[0], lrg_result[0].intercept + lrg_result[0].slope * trimmed_freq[0], color='k',
                 linestyle='dashed')
        plt.plot(trimmed_freq[1], lrg_result[1].intercept + lrg_result[1].slope * trimmed_freq[1], color='k',
                 linestyle='dashed')
        # plt.axvline(fc+6*linewidth)
        # plt.axvline(fc-6*linewidth)
        plt.ylabel('Magnitude (dB)')
        plt.xlabel('Frequency')
        plt.show()

    new_sdata = remove_linear_sigma(fdata, sdata, avg_sigma)

    return new_sdata/np.max(np.abs(new_sdata))

def normalize(fdata, sdata, anchor_point = 1+1j*0):
    orpmodel = lmfit.Model(sloped_arctan)
    #from this we just need fr and approximate Ql
    filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
    gradS = np.gradient(filtered_data, fdata)
    gradSmagnitude = np.abs(gradS)
    # estimate center frequency
    fc_index = np.argmax(gradSmagnitude)
    fc = fdata[fc_index]
    # estimate linewidth -- count up the spans of adjacent frequency points that are close to resonance
    chiFunction = partitionFrequencyBand(fdata, gradS)
    linewidth = np.dot(chiFunction[:-1], np.diff(fdata))
    #DO the circlefit
    xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
    zc = xc + 1j*yc


    orparams = lmfit.Parameters()
    orparams.add(name='Ql', value=fc/linewidth)
    orparams.add(name='fr', value=fc)
    orparams.add(name='delay', value=0, vary=False)
    orparams.add(name='theta_0', value=0)


    #the sdata needs to be centered first
    orp_results = orpmodel.fit(np.unwrap(np.angle(sdata-zc)), orparams, f=fdata)
    # plt.plot(fdata, np.unwrap(np.angle(sdata-zc)))
    # plt.plot(fdata, orpmodel.func(fdata, orparams['Ql'], orparams['fr'], orparams['delay'], orparams['theta_0']))
    # plt.show()
    theta_0 = orp_results.params['theta_0'].value
    beta = fmod(theta_0 + np.pi, np.pi)

    orp =  xc + 1j * yc + r * np.exp(1j * beta)
    return anchor_point*sdata/orp

def preprocess(fdata, sdata, **kwargs):
    do_linear_preprocessing = kwargs.get('linear_preprocess', False)
    delay_preprocessing = kwargs.get('preprocess_delay', True)
    normalize_data = kwargs.get('normalize', True)
    anchor_point = kwargs.get('anchor_point', 1+0*1j)

    if do_linear_preprocessing:
        sdata = preprocess_linear(fdata, sdata)
    if delay_preprocessing:
        sdata = preprocess_delay(fdata, sdata)
    if normalize_data:
        sdata = normalize(fdata, sdata, anchor_point)

    return sdata



#TODO: add an hpspace function to generate homophasal spacing of frequency points