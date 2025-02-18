from abc import ABC, abstractmethod
import numpy as np
import lmfit

class FitMethod(ABC):
    @abstractmethod
    def func(self, x: np.ndarray, *params) -> np.ndarray:
        """
        The fit function for the method.

        Args:
            x (np.ndarray): The independent variable data.
            *params: The parameters for the fit function.

        Returns:
            np.ndarray: The evaluated fit function.
        """
        pass

    @abstractmethod
    def find_initial_guess(self, x: np.ndarray, y: np.ndarray) -> lmfit.Parameters:
        """
        Finds an initial guess for the parameters based on the data.

        Args:
            x (np.ndarray): The independent variable data.
            y (np.ndarray): The dependent variable data.

        Returns:
            Parameters: The initial guess for the parameters
        """
        pass

    @abstractmethod
    def create_model(self) -> lmfit.Model:
        """
        Creates an lmfit model of the method function

        Returns:
            Model: object with minimizing methods
        """
        pass

    #this method should be overwritten for conformal mapping methods
    @abstractmethod
    def fit_procedure(self, fdata, sdata, params):
        model = self.create_model(self)#idk why but self needs to be passed explicitly here
        result = model.fit(sdata, params, f=fdata, method='leastsq')
        #TODO: should run the confidence interval finder at this point
        return result, []

    @abstractmethod
    def off_res_point(self, params):
        return 1+0*1j
