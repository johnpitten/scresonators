import lmfit.models
import numpy as np
from ..utils import *
from ..plotting.plotres import *
import matplotlib.pyplot as plt
from .fit_method import FitMethod
from abc import ABC, abstractmethod
from .dcm import DCM
from ..utils import find_circle
from scipy.ndimage import gaussian_filter

class CM_DCM(DCM):
    """
    Conformal Mapping variant of DCM

    Cliff Chen, David Perello, Shahriar Aghaeimeibodi, Guillaume Marcaud, Ignace Jarrige, Hanho Lee, Warren Fon,
    Matt Matheny, Jiansong Gao; Efficient methods for extracting superconducting resonator loss in
    the single-photon regime. J. Appl. Phys. 28 January 2025; 137 (4): 044401. https://doi.org/10.1063/5.0242201

    Everything but the fit_procedure() method remains the same
    """
    def __init__(self):
        pass

    @staticmethod
    def w_func(Im_w, phi, b):
        return np.tan(phi)*Im_w + b

    @staticmethod
    def w1_func(f, f0, Q):
        return 2*Q*(f-f0)/f0

    @abstractmethod
    def fit_procedure(self, fdata, sdata, params):
        '''
        The fit procedure for the conformal mapping variant of the
        '''

        w = 1/(1-sdata)

        Q = params['Q'].value
        Qc = params['Qc'].value
        phi = params['phi'].value
        f0 = params['f0'].value

        w_params = lmfit.Parameters()
        w_params.add(name = 'phi', value = phi)
        w_params.add(name = 'b', value = Qc/(Q*np.cos(phi)))#this initial value may need to be changed

        w_model = lmfit.Model(self.w_func, independent_vars=['Im_w'])
        w_results = w_model.fit(np.real(w), w_params, Im_w = np.imag(w), method = 'leastsq')

        phi = w_results.params['phi'].value
        params['phi'] = w_results.params['phi']

        w1 = np.exp(1j*phi)*w

        w1_params = lmfit.Parameters()
        w1_params.add(name = 'f0', value = f0)
        w1_params.add(name = 'Q', value = Q)

        w1_model = lmfit.Model(self.w1_func, independent_vars=['f'])
        w1_results = w1_model.fit(np.imag(w1)/np.real(w1), w1_params, f = fdata)

        f0 = w1_results.params['f0']
        Q = w1_results.params['Q']
        params['f0'] = w1_results.params['f0']
        params['Q'] = w1_results.params['Q']

        #Source paper is unclear how to obtain Qc in a way that also gives an uncertainty
        Qc = np.average(np.real(w1))*Q
        # John Taylor's Error Analysis p. 32
        Qc_std = Qc*(np.std(np.real(w1))/np.average(np.real(w1)) + params['Q'].stderr/Q)

        params['Qc'].value = Qc
        params['Qc'].stderr = Qc_std

        w2 = (w1-np.cos(phi))/Qc

        #TODO: determine how to propagate uncertainties between data transformations
        invQi = np.average(np.real(w2))
        invQi_std = np.std(np.real(w2))

        params.add(name = 'invQi', value = invQi)
        params['invQi'].stderr = invQi_std
        #Create a ModelResult object to store results & evaluate for plotting
        model = self.create_model(self)
        fit_result = lmfit.model.ModelResult(model = model, params = params)
        fit_result.userkws = {'f': fdata}#this is how lmfit stores the independent data internally

        return fit_result, [w_results, w1_results]


    def extractQi(self, params):
        #TODO: add calculation for Qi and its std_err -- for now invQi will do
        return params

    @staticmethod
    def makePanel():
        fig, ax = plt.subplot_mosaic([['w','w1','w2']], layout = 'constrained')

        ax['w'].set_aspect('equal')
        ax['w'].set_title('$w$-Plane'+'\n'+r'$w = 1/(1-S)$')
        ax['w'].set_xlabel(r'Re$[w]$')
        ax['w'].set_ylabel(r'Im$[w]$')

        ax['w1'].set_title(r'$w_1$-Plane'+'\n'+r'$w_1 = e^{j\phi}w$')
        ax['w1'].set_xlabel('Frequency (GHz)')
        ax['w1'].set_ylabel(r'$\frac{\text{Im}[w_1]}{\text{Re}[w_1]}$')


        ax['w2'].set_title(r'$w_2$-Plane'+'\n'+r'$w_2 = Q_i^{-1}+2j\,\delta f / f_0$')
        ax['w2'].set_xlabel(r'Re$[w_2]$')
        ax['w2'].set_ylabel(r'Im$[w_2]$')
        return fig, ax


    @staticmethod
    def plot_w_planes(fdata: np.array, sdata: np.array, result_list: list):
        fig = plt.gcf()
        ax_list = fig.get_axes()
        ax = AxesListToDict(ax_list)

        w_results = result_list[0]
        w1_results = result_list[1]
        w = 1/(1-preprocess(fdata, sdata))
        phi = w_results.params['phi'].value
        w1 = np.exp(1j*phi)*w
        Q = w1_results.params['Q']
        Qc = np.average(np.real(w1))*Q
        w2 = (w1 - np.cos(phi)) / Qc

        ax['w'].plot(np.real(w), np.imag(w), color = 'b', label = 'data')
        ax['w'].plot(w_results.eval(), w_results.userkws['Im_w'], color = 'k', linestyle = 'dashed', label = 'fit')

        ax['w1'].plot(fdata, np.imag(w1)/np.real(w1), color = 'b', label = 'data')
        ax['w1'].plot(w1_results.userkws['f'], w1_results.eval(), color = 'k', linestyle = 'dashed', label = 'fit')

        ax['w2'].plot(np.real(w2), np.imag(w2), label = 'data', color = 'b')
        ax['w2'].axvline(x=np.average(np.real(w2)), label = r'Re$[w_2]$ average', color = 'k', linestyle = 'dashed')

        return fig, ax

