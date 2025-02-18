from .hanger import HangerMode
from .reflection import ReflectionMode
from .dcm import DCM
from .CM_DCM import CM_DCM
fit_methods = dict()

fit_methods['Reflection'] = ReflectionMode
fit_methods['Refl'] = ReflectionMode
fit_methods['Refl.'] = ReflectionMode
fit_methods['Reflect'] = ReflectionMode

fit_methods['Hanger'] = HangerMode

fit_methods['DCM'] = DCM

fit_methods['CM-DCM'] = CM_DCM
fit_methods['DCM-CM'] = CM_DCM