'''controllers for optimization routines in Python'''

from collections import OrderedDict
import functools
from scipy.optimize import minimize
from loggers import MemoryLogger, NullLogger

class SimpleMinimizer(object):
    '''interactive wrapper for scipy.optimize.minimize'''

    def __init__(self, func, parameters, fargs, **kwargs):

        init = [p.value_ for p in parameters.flatten(freeonly=True)]

        self.parameters = parameters

        self.args = OrderedDict([('method', kwargs.get('method', 'Powell')),
            ('tol', kwargs.get('tol', 1e-3)), ('x0', init)])
        
        self.logger = NullLogger(self.args, parameters)

        self.func = CriterionFunction(func, parameters, fargs, self.logger)

    def log(self, mode='memory'):
        '''log the optimization

        Parameters
        ----------
        log : bool
            True to log, False to not log
        mode : str
            'memory' to log within memory, 'disk' to log on
            disk
        '''

        if mode == 'memory':
            self.logger = MemoryLogger(self.args, self.parameters)
        elif mode == 'disk':
            self.logger = DiskLogger(self.args, self.parameters)
        elif mode == 'null':
            self.logger = NullLogger(self.args, self.parameters)
        else:
            raise ValueError('mode {} must be "memory", "disk", or "null"')

        self.func.logger = self.logger

    def run(self):
        '''Run the optimizer given the current parameter values'''

        try:
            self.results = minimize(self.func, **self.args)
        except KeyboardInterrupt:
            pass

class CriterionFunction(object):
    '''criterion function to be optimized

    ** Attributes **
    func : callable
        the criterion function, should take parameters
        as they were entered into the parameter space as
        first arguments, `fargs` will be tacked on to the
        end
    parameters : ParameterSpace instance
        contains the parameters to be updated and fed into
        the criterion function as the first arguments
    fargs : tuple
        second set of arguments to go into criterion function
    '''

    def __init__(self, func, parameters, fargs, logger):
        self.func = func
        self.parameters = parameters
        self.fargs = fargs
        self.logger = logger

    def __call__(self, values):

        self.parameters.update(values)

        params = tuple(p.value for p in self.parameters)

        fval = self.logger(functools.partial(self.func, *(params + self.fargs)))

        return fval