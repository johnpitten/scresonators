import numpy as np
import logging
import lmfit
import scipy.optimize as spopt
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d
from .utils import *
from math import fmod
from warnings import warn
import skrf as rf

#TODO: Can't we just have the Fitter inherit all the methods of the fit method class?

class Fitter:
    def __init__(self, fit_method=None, **kwargs):
        """Initializes the Fitter with a fitting method that includes the fitting function.

        Args:
            fit_method (object): An instance of a fitting method class that contains the `func` method.
        """
        if fit_method is None or not hasattr(fit_method, 'func'):
            raise ValueError("A fitting method with a valid 'func' attribute must be provided.")
        #TODO: kwargs should be passed to Resonator.fit(), not Resonator.set_fitting_strategy()
        self.fit_method = fit_method
        self.remove_elec_delay = kwargs.get('remove_delay', True)
        self.preprocess_circle = kwargs.get('preprocess_circle', True)
        self.preprocess_linear = kwargs.get('preprocess_linear', False)
        self.databg = kwargs.get('databg', None)#remove?
        self.plot_results = kwargs.get('plotstyle', None)#for later implementation of optional plotting post fitting
        self.delay_guess = kwargs.get('delay_guess', None)
        self.Ql_guess = None
        self.fr_guess = None
        self.theta_0 = None
        self.phi = None
        self.off_res_point = kwargs.get('off_res_point', 1+0*1j)
        self.init_params = None
        self.fitResult = None

    def fit(self, fdata, sdata, manual_init=None, verbose=False) -> lmfit.model.ModelResult:
        """Fit resonator data using the provided method and lmfit's Model fit"""
        #fdata: numpy array of the frequency data
        #sdata: complex valued numpy array of the scattering parameter data


        ##########################################
        #PREPROCESSING
        ##########################################
        if self.databg:
            #this feature is untested
            sdata = self.background_removal(sdata)
        if self.preprocess_linear == True:
            sdata = preprocess_linear(fdata, sdata)
            #TODO: need better variable names
        if self.remove_elec_delay == True:
            #TODO: replace with the utils function
            delay = self.initial_guess_delay(fdata, sdata)
            sdata = remove_delay(fdata, sdata, delay)
        if self.preprocess_circle == True:
            #rotate and scale the off-resonant point to a prescribed anchor point
            sdata = self.anchor_to_point(fdata, sdata)

        ######################################
        #Initial guess for fitting parameters
        ######################################
        # Setup the initial parameters or use provided manual_init
        if manual_init:
            params = manual_init
        else:
            #very weird that self = self.fit_method needs to be passed
            params = self.fit_method.find_initial_guess(self = self.fit_method, fdata = fdata, sdata = sdata)
            self.init_params = params

        #####################################################
        #The actual fit, implemented with the lmfit package
        #####################################################

        #TODO: fit_procedure() needs to return a list or dict of ModelResults if there were multiple fits
        #TODO: helper function which grabs the resonator params from the list or dict
        result, intermediate_results = self.fit_method.fit_procedure(self = self.fit_method, fdata = fdata, sdata = sdata, params = params)
        #if verbose: print(result.fit_report())
        self.fitResult = result


        return result, intermediate_results

    def background_removal(self, linear_amps: np.ndarray, phases: np.ndarray):
        """
        Removes background signal by interpolating and adjusting amplitude and phase,
        using stored background data.

        Untested

        Args:
            linear_amps (np.ndarray): Measured linear amplitudes to be corrected.
            phases (np.ndarray): Measured phases to be corrected.

        Returns:
            np.ndarray: Corrected complex S21 data with background removed.
        """
        if not self.databg:
            raise ValueError("Background data ('databg') not provided.")

        # Extract background data
        x_bg = self.databg.freqs
        linear_amps_bg = self.databg.linear_amps
        phases_bg = self.databg.phases

        # Create interpolation functions for background amplitude and phase
        fmag = interp1d(x_bg, linear_amps_bg, kind='cubic', fill_value="extrapolate")
        fang = interp1d(x_bg, phases_bg, kind='cubic', fill_value="extrapolate")

        # Correct measured data using interpolated background
        linear_amps_corrected = np.divide(linear_amps, fmag(self.databg.freqs))
        phases_corrected = np.subtract(phases, fang(self.databg.freqs))

        # Return corrected data as complex S21 values
        return np.multiply(linear_amps_corrected, np.exp(1j * phases_corrected))


    def find_delay(self, fdata: np.ndarray, sdata: np.ndarray):
        """
        Modified version of the Criclefit method described in Probst.

        Adjusts electrical delay such that the offset phase most closely fits an arctangent -- works poorly

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        """
        params = lmfit.Parameters()
        guess_params = self.fit_method.find_initial_guess(self = self.fit_method ,fdata = fdata,sdata = sdata)

        if self.delay_guess == None:
            delay_guess = self.initial_guess_delay(fdata, sdata)/1.25
        elif self.delay_guess == 0:
            delay_guess = 1e-16*self.guess_delay(fdata, sdata)
        else:
            delay_guess = self.delay_guess


        params.add(name = 'fr', value = guess_params['f0'].value)
        params.add(name = 'Ql', value = guess_params['Q'].value)
        params.add(name = 'delay', value = delay_guess)
        params.add(name = 'theta_0', value = 0)


        #running the minimizer several times also improves results
        for iter in range(30):
            #alternate running the least squares minimization & brute force only varying the delay?
            min_result = lmfit.minimize(fcn=self.arctan_deviation, params=params, args=(fdata, sdata),
                                    method='leastsq', max_nfev = 10000000)

            params = min_result.params

            #TODO: we should have a condition on the residuals to break this loop early
        electrical_delay = params['delay'].value
        self.Ql_guess = params['Ql'].value
        self.fr_guess = params['fr'].value
        self.theta_0 = params['theta_0'].value

        #Plot the sloped arctan as a verification step
        '''
        import matplotlib.pyplot as plt
        sdata_new = remove_delay(fdata, sdata, electrical_delay)
        xc, yc, r = find_circle(np.real(sdata_new), np.imag(sdata_new))

        offset_phase = params['theta_0'].value+2*np.arctan(2*params['Ql'].value*(1-fdata/params['fr'].value))

        plt.plot(fdata, np.unwrap(np.angle(sdata_new-(xc+1j*yc))), label = 'data')
        plt.plot(fdata, offset_phase, label = 'min fit')
        plt.ylabel('offset phase (rad.)')
        plt.xlabel('frequency (a.u.)')
        plt.legend()
        plt.show()
        '''

        return electrical_delay

    def guess_delay(self, fdata: np.ndarray, sdata: np.ndarray):
        """
        Linear fit of the phase to be used as an initial guess of the electrical delay.

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        """
        phase = np.unwrap(np.angle(sdata))
        lrg_result = linregress(fdata, phase)
        delay_guess = lrg_result.slope/(-2*np.pi)

        return delay_guess


    def initial_guess_delay(self, fdata: np.ndarray, sdata: np.ndarray):
        '''
        Just fit the phase while discarding data in the linewidth.

        Messy -- needs cleaning up -JP

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        '''
        filtered_data = gaussian_filter(sdata, sigma=3)  # sigma may need to be changed for noisy data
        gradS = np.gradient(filtered_data, fdata)
        gradSmagnitude = np.abs(gradS)
        fc_index = np.argmax(gradSmagnitude)
        fc = fdata[fc_index]
        #TODO: instead of using a cutoff, first estimate the linewidth, then discard data within n linewidths
        #or simply identify f0, and discard data within fspan/2 of f0
        guess_params = self.fit_method.find_initial_guess(self = self.fit_method ,fdata = fdata,sdata = sdata)
        linewidth = 2*fc/guess_params['Q']

        chiFunction = np.zeros(len(fdata))
        for n in range(len(chiFunction)):
            if np.abs(fdata[n]-fc) < 6*linewidth:
                chiFunction[n] = 1


        freq_arrays = np.split(fdata, [fc_index])
        data_arrays = np.split(sdata, [fc_index])
        chi_arrays = np.split(chiFunction, [fc_index])

        #TODO: trim data and perform two separate fits, average values for delay_guess

        delay_fit = []
        lrg_result = [0,0]
        trimmed_freq = [0,0]
        for n in range(2):
            #trim the data
            trim = np.nonzero(chi_arrays[n])
            trimmed_freq[n] = np.delete(freq_arrays[n], trim)
            trimmed_data = np.delete(data_arrays[n], trim)

            #fit trimmed freq and data
            #TODO: check the r**2 of these fits and warn the user if they're below 0.9
            trimmed_phase = np.unwrap(np.angle(trimmed_data))
            lrg_result[n] = linregress(trimmed_freq[n], trimmed_phase)
            delay_guess = lrg_result[n].slope / (-2 * np.pi)
            delay_fit = np.append(delay_fit, delay_guess)
        avg_delay_guess = (delay_fit[0]+delay_fit[1])/2
        #warm user if r^2 <0.9
        for n in range(2):
            if lrg_result[n].rvalue**2 < 0.9:
                warn(f'low r-squared in delay fit: {lrg_result[n].rvalue**2}')

        #plot the fits as a check
        '''
        import matplotlib.pyplot as plt
        plt.plot(fdata, np.unwrap(np.angle(sdata)))
        plt.plot(trimmed_freq[0], lrg_result[0].intercept+lrg_result[0].slope*trimmed_freq[0], color = 'k', linestyle = 'dashed')
        plt.plot(trimmed_freq[1], lrg_result[1].intercept + lrg_result[1].slope * trimmed_freq[1], color='k',
                 linestyle='dashed')
        #plt.axvline(fc+6*linewidth)
        #plt.axvline(fc-6*linewidth)
        plt.ylabel('Phase (rad.)')
        plt.xlabel('Frequency')
        plt.show()
        '''


        #print(f'initial delay guess: {avg_delay_guess}')
        return avg_delay_guess

    #TODO: needs testing
    def find_linear_sigma(self, fdata: np.ndarray, sdata: np.ndarray):
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
        fc_index = np.argmax(gradSmagnitude)
        fc = fdata[fc_index]
        #TODO: instead of using a cutoff, first estimate the linewidth, then discard data within n linewidths
        #or simply identify f0, and discard data within fspan/2 of f0
        guess_params = self.fit_method.find_initial_guess(self = self.fit_method ,fdata = fdata,sdata = sdata)
        linewidth = 2*fc/guess_params['Q']

        chiFunction = np.zeros(len(fdata))
        for n in range(len(chiFunction)):
            if np.abs(fdata[n]-fc) < 6*linewidth:
                chiFunction[n] = 1


        freq_arrays = np.split(fdata, [fc_index])
        data_arrays = np.split(sdata, [fc_index])
        chi_arrays = np.split(chiFunction, [fc_index])

        #TODO: trim data and perform two separate fits, average values for delay_guess

        sigma_fit = []
        lrg_result = [0,0]
        trimmed_freq = [0,0]
        for n in range(2):
            #trim the data
            trim = np.nonzero(chi_arrays[n])
            trimmed_freq[n] = np.delete(freq_arrays[n], trim)
            trimmed_data = np.delete(data_arrays[n], trim)

            #fit trimmed freq and data
            #TODO: check the r**2 of these fits and warn the user if they're below 0.9
            trimmed_mag = 20*np.log10(np.abs(trimmed_data))
            lrg_result[n] = linregress(trimmed_freq[n], trimmed_mag)
            sigma = -np.log(10) * lrg_result[n].slope / (40*np.pi)
            sigma_fit = np.append(sigma_fit, sigma)
        avg_sigma = (sigma_fit[0]+sigma_fit[1])/2
        #warm user if r^2 <0.9
        for n in range(2):
            if lrg_result[n].rvalue**2 < 0.9:
                warn(f'low r-squared in delay fit: {lrg_result[n].rvalue**2}')

        #plot the fits as a check
        '''
        import matplotlib.pyplot as plt
        plt.plot(fdata, 20*np.log10(np.abs(sdata)))
        plt.plot(trimmed_freq[0], lrg_result[0].intercept+lrg_result[0].slope*trimmed_freq[0], color = 'k', linestyle = 'dashed')
        plt.plot(trimmed_freq[1], lrg_result[1].intercept + lrg_result[1].slope * trimmed_freq[1], color='k',
                 linestyle='dashed')
        #plt.axvline(fc+6*linewidth)
        #plt.axvline(fc-6*linewidth)
        plt.ylabel('Magnitude (dB)')
        plt.xlabel('Frequency')
        plt.show()
        '''


        #print(f'initial delay guess: {avg_delay_guess}')
        return avg_sigma

    def find_delay_circlefit(self, fdata: np.ndarray, sdata: np.ndarray):
        '''
        Finds the electrical delay using the circle fit method described in Probst et al. -- works poorly

        The delay is varied to minimize the deviation of the scattering data from an ideal circle in the complex plane.
        REVIEW OF SCIENTIFIC INSTRUMENTS 86, 024706 (2015). Results are unsatisfactory, but may be improved by adopting
        the heuristics present in self.find_delay

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            delay: float representing the electrical delay
        '''

        phase_data = np.unwrap(np.angle(sdata))
        #the initial guess just needs to get the scale right, this should be good enough
        delay_guess = self.guess_delay(fdata, sdata)
        #print(f'delay initial guess: {delay_guess}')

        # make an lmfit.Parameters object and add a single lmfit.Parameter object representing the electrical delay
        delay_params = lmfit.Parameters()
        delay_params.add('delay', delay_guess, max = delay_guess+0.2*np.abs(delay_guess), min = delay_guess-0.2*np.abs(delay_guess), brute_step = np.abs(delay_guess)/10000)

        min_result = lmfit.minimize(fcn = self.circle_deviation, params = delay_params, args = (fdata, sdata), method = 'leastsq')
        electrical_delay = min_result.params['delay'].value

        return electrical_delay

    def arctan_deviation(self, params: lmfit.Parameters, fdata: np.ndarray, sdata: np.ndarray):
        '''
        An objective function to be minimized to find the electrical delay.

        The delay value passed through params is removed from the scattering data. Then the best fit circle is found for
        the resulting data. Its center is translated to the origin. If the delay removed corresponds to the actual
        electrical delay then the phase of this transformed data is an arctangent.

        Args:
            params: lmfit.Parameters object to store parameters for the arctangent function and the delay
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            residuals: numpy array of the residuals comparing the transformed data to the arctangent described by params
        '''
        fr = params['fr'].value
        Ql = params['Ql'].value
        t_delay = params['delay'].value
        theta_0 = params['theta_0'].value

        sdata = remove_delay(fdata, sdata, t_delay)
        xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
        sdata = sdata - (xc+1j*yc)

        offset_phase = np.unwrap(np.angle(sdata))

        residuals = np.abs(offset_phase - (theta_0+2*np.arctan(2*Ql*(1-fdata/fr))))
        return residuals

    def circle_deviation(self, params: lmfit.Parameters, fdata: np.ndarray, sdata: np.ndarray):
        '''
        this is the objective function to be minimized to find the electrical delay in the Probst circlefit method
        '''
        # the signature must be: fcn(params, *args, **kws)
        # the data is passed into minimize with args = (fdata, sdata) as a tuple
        # electrical delay is the only param, but the data is passed through *args
        # this must return an array of residuals r^2-(x_n-x_c)^2-(y_n-y_c)^2

        # unpack param value
        delay = params['delay'].value
        # remove that amount of delay
        adjusted_sdata = remove_delay(fdata, sdata, delay)
        # grab x any y components of sdata
        # calculate r, x_c, and y_c from the circlefit method
        x_c, y_c, r = find_circle(np.real(sdata), np.imag(sdata))
        # calculate and return array of terms like r**2-(x_n-x_c)**2-(y_n-y_c)**2
        return np.abs(r**2-(np.real(adjusted_sdata)-x_c)**2-(np.imag(adjusted_sdata)-y_c)**2)

    def find_off_res_point(self, fdata, sdata):
        """
        This replaces the poorly named calibration function
        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data

        Returns:
            the off-resonant point as a complex float
            additionally sets self.phi
        """
        if self.theta_0 is not None:
            beta = fmod(self.theta_0+np.pi, np.pi)
            xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
            self.phi = fmod(np.pi - self.theta_0 + np.angle(xc+1j*yc), np.pi)
            #print(f'phi (preprocessing): {self.phi}')
            return xc+ 1j*yc+ r*np.exp(1j*beta)
        else:
            #this block is untested
            #TODO: fit the offset phase to an arctan (without delay, that is not the place for this function)
            xc, yc, r = find_circle(np.real(sdata), np.imag(sdata))
            zc = xc+1j*yc
            orpmodel = lmfit.Model(sloped_arctan)
            guess_params = self.fit_method.find_initial_guess(self=self.fit_method, fdata=fdata, sdata=sdata)

            orparams = lmfit.Parameters()
            orparams.add(name = 'Ql', value = guess_params['Q'].value)
            orparams.add(name = 'fr', value = guess_params['f0'].value)
            orparams.add(name = 'delay', value = 0, vary = False)
            orparams.add(name = 'theta_0', value = 0)

            orp_results = orpmodel.fit(np.unwrap(np.angle(sdata-zc)), orparams, f=fdata)
            theta_0 = orp_results.params['theta_0'].value
            beta = fmod(theta_0 + np.pi, np.pi)

            self.phi = fmod(np.pi - theta_0 + np.angle(xc + 1j * yc), np.pi)
            #print(f'phi (preprocessing): {self.phi}')
            return xc + 1j * yc + r * np.exp(1j * beta)

    def anchor_to_point(self, fdata: np.ndarray, sdata: np.ndarray, anchor_point = None):
        """
        Rotates and scales the scattering data to send the off-resonant point to an anchor point (e.g. 1+0j).

        Args:
            fdata: numpy array of the frequency data
            sdata: numpy array of the scattering data
            anchor_point: optional complex float to specify the desired off-resonant point (defaults to 1+0j)

        Returns:
            sdata: numpy array of the transformed scattering data
        """
        if anchor_point == None:
            anchor_point = self.off_res_point

        Old_ORP = self.find_off_res_point(fdata, sdata)
        sdata = anchor_point*sdata/Old_ORP
        return sdata
