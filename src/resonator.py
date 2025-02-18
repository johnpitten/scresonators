from types import NoneType


from skrf import Network
from .fitter import Fitter
from .fit_methods.fit_method import FitMethod
from .plotting import plotres
import matplotlib.pyplot as plt
from .utils import *
from .fit_methods.methods import fit_methods



"""
only take data in appropriately shaped np arrays
fdata: np shape (,M)
sdata: np shape (M,N,N) for an N-Port device with M frequency points, shape (,M) also works for a 1-Port device 
this removes ambiguity from amplitude/phase data (linear or db & degrees or radians)
and naturally generalizes for Effective Reflection Mode measurements (Two Port required)
"""

class Resonator:
    def __init__(self, fdata=None, sdata = None, strategy = None, **kwargs):
        self.fdata = fdata
        self.sdata = sdata
        self.fitter = None
        self.fit_params = None
        self.fitResult = None
        self.intermediateResults = None

        if strategy:
            self.set_fitting_strategy(strategy = strategy, **kwargs)



    # TODO: Can we use *args to combine the following three functions into one?
    def load_data(self, fdata, sdata):
        """Load data into the resonator object."""
        self.fdata = fdata
        self.sdata = sdata

    def Load_data_from_Network(self, network: Network):
        '''load data from an rf.Network object'''
        #TODO: check fitmethod to decide whether to throw out S11, S22
        self.sdata = network.s
        network.frequency.units = 'GHz'
        self.fdata = network.f


    def load_data_from_touchstone(self, touchstone_file: str):
        network = Network(touchstone_file)
        self.sdata = network.s
        network.frequency.units = 'GHz'
        self.fdata = network.f_scaled

    #TODO: kwargs should be passed through Resonator.fit instead
    def set_fitting_strategy(self, strategy: FitMethod | str, **kwargs):
        """Set the fitting strategy with a FitMethod object."""

        if type(strategy) == str:
            self.fitter = Fitter(fit_method = fit_methods[strategy], **kwargs)
        else:
            self.fitter = Fitter(fit_method=strategy, **kwargs)

    def fit(self, manual_init=None, verbose = False,):
        #TODO: introduce **kwargs to pass onto fitter.fit(), also specify if plotting is wanted post fit.
        """Perform fitting using the selected fitting strategy."""
        if not self.fitter:
            raise ValueError("Fitting strategy not set.")
        if type(self.fdata) == NoneType or type(self.sdata) == NoneType:
            raise ValueError("Data not loaded")

        result, intermed_results = self.fitter.fit(self.fdata, self.sdata, manual_init=manual_init, verbose = verbose)
        #TODO: make Fitter.fit return ModelResult instead of Parameters
        self.fitResult = result
        self.intermediateResults = intermed_results

        params = self.fitter.fit_method.extractQi(self=self.fitter.fit_method, params=result.params)
        #TODO: change to using ModelResult.conf_interval() and ModelResult.ci_out() to calculate conf. intervals
        params = std_err_to_95pct(params)
        self.fit_params = params
        return params

    def plotSummary(self, display = True):
        '''
        Generate and display Summary plot using fit_params, only do this after fitting
        '''
        fdata = self.fdata
        sdata = self.sdata
        fit_params = self.fit_params

        anchor_point = self.fitter.fit_method.off_res_point(self = self.fitter.fit_method, params = fit_params)


        fig, ax = plotres.makeSummaryFigure()
        plotres.summaryPlot(fdata, preprocess(fdata, sdata, anchor_point = anchor_point), color='blue', label='data')
        plotres.summaryPlot(fdata, anchor_point*self.fitResult.eval(),
                            color='k', linestyle='dashed', label='fit')

        #plotres.displayAllParams(self.fit_params)
        if display == True:
            plt.show()
        return fig, ax

    def annotateRedchi(self):
        #grab fig, ax
        fig = plt.gcf()
        ax_list = fig.get_axes()
        ax = plotres.AxesListToDict(ax_list)
        #grab reduced ch squared value
        redchi = self.fitResult.redchi
        #round appropriately
        formatted_redchi = "{:.3e}".format(redchi)
        #construct annotation string
        annotation_string = '\n'+r'$\chi_\nu^2 = $' + formatted_redchi
        #append to annotation
        plotres.annotate(annotation_string)
        pass

    def plotIntermediateResults(self):
        #call function defined in e.g. CM-DCM to generate subplot mosaic of w, w1, w2 planes,
        # then another to populate it with the results
        self.fitter.fit_method.makePanel()
        fig, ax = self.fitter.fit_method.plot_w_planes(fdata = self.fdata, sdata = self.sdata,
                                             result_list = self.intermediateResults)
        return fig, ax

