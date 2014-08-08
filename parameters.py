'''containers for parameters used in optimization'''

from __future__ import print_function, division

# standard library
import itertools
from collections import OrderedDict

# third party
import numpy as np

class ParameterBase(object):
    '''Base parameter class
    
    ** Attributes **
    
    '''
    
    def __init__(self, value, name):
        self.name = name
        self.value = value
        self.value_ = value
    
    def set_bounds(self):
        '''sets bounds of parameter'''
        raise NotImplementedError
    
    def set_free(self):
        '''distinguishes parameter as free (not fixed)'''
        raise NotImplementedError
    
    def update(self):
        '''updates value'''
        raise NotImplementedError
        
    def __str__(self):
        return self.value.__str__()

    def __repr__(self):
        return self.value.__repr__()

class ParameterScalar(ParameterBase):
    '''parameter structure for scalar parameters
    
    ** Attributes **
    
    value : scalar
        internal (i.e. constrained) parameter value, to
        be used in function evaluation, conforms to bounds,
        same as external value if parameter is unbounded
    value_ : scalar
        external (i.e. unconstrained) parameter value, to
        be used in function evaluation, freely various in 
        real space, same as internal value if parameter is
        unbounded
    bounds : tuple
        length 2 tuple of scalars containing minimum and 
        maximum value of parameter
    free : bool
        whether parameter is free to vary
    '''
    
    def __init__(self, value, name, free=True, bounds=(-np.inf, np.inf)):
        ParameterBase.__init__(self, value, name)
        self.free = free
        self.set_bounds(*bounds)

    @property
    def flat(self):
        return iter([self])

    def copy(self):
        return ParameterScalar(self.value, self.name, self.free, self.bounds)

    def set_bounds(self, min_, max_):
        '''set bounds of parameter
        
        Parameters
        ----------
        min_ : numeric
            minimum bound for parameter
        max_ : numeric
            maximum bound for parameter
        '''

        if (self.value > max_) | (self.value < min_):
            raise ValueError('Bounds conflict with parameter values')

        self.bounds = (min_, max_)

        if ((not np.isfinite(min_)) & (not np.isfinite(max_))) | (not self.free):
            self.to_internal = lambda val: val
            self.to_external = lambda val: val
        elif np.isfinite(min_):
            self.to_external = lambda val: np.sqrt((val - min_ + 1)**2 - 1)
            self.to_internal = lambda val: min_ - 1 + np.sqrt(val**2 + 1)
        elif np.isfinite(max_):
            self.to_external = lambda val: np.sqrt((max_ - val + 1)**2 - 1)
            self.to_internal = lambda val: max_ + 1 - np.sqrt(val**2 + 1)
        else:
            self.to_external = lambda val: np.arcsin(2*(val - min_)/(max_ - min_) - 1)
            self.to_internal = lambda val: min_ + (np.sin(val) + 1)*(max_ - min_)/2

        self.value_ = self.to_external(self.value)
    
    def set_free(self, free):
        '''set the parameter as free
        
        Parameters
        ----------
        free : bool
            True to free parameter, False otherwise
        '''
        
        self.free = bool(free)
        self.set_bounds(*self.bounds)

    def update(self, value, source='external'):
        '''updates value of parameter
        
        Parameters
        ----------
        value : numeric type
            value to update with
        source : str
            'external' to update with an unconstrained value,
            'internal' to update with a constrained value
        '''

        assert self.free
        assert source in ['external', 'internal']

        if source == 'external':
            self.value = self.to_internal(value)
            self.value_ = value
        else:
            self.value = value
            self.value_ = self.to_external(value)
            
    def summary(self):
        '''print summary of the parameter'''
        
        string = '{:<10}{}\n'.format('Name:', self.name)
        string += '{:<10}{}\n'.format('Value:', self.value)
        string += '{:<10}{}\n'.format('Bounds:', self.bounds)
        string += '{:<10}{}\n'.format('Free:', self.free)
        print(string)

class ParameterArray(object):
    '''class to contain array of parameters'''
    
    def __init__(self, scalars):
        self.scalars = scalars

    def copy(self):
        return np.reshape([p.copy() for p in self.scalars.flat], self.scalars.shape)

    @property
    def flat(self):
        return self.scalars.flat

    @property
    def value(self):
        return np.resize([p.value for p in self.scalars.flat], self.scalars.shape)
    
    @property
    def value_(self):
        return np.resize([p.value_ for p in self.scalars.flat], self.scalars.shape)

    def set_bounds(self, min_, max_):
        '''sets bounds of parameter
        
        Parameters
        ----------
        min_ : numeric or numeric array
            minimum bound for each parameter in array
        max_ : numeric or numeric array
            maximum bound for each parameter in array
        '''
        min_ = np.resize(min_, self.scalars.size)
        max_ = np.resize(max_, self.scalars.size)

        for i,param in enumerate(self.scalars.flat):
            param.set_bounds(min_[i], max_[i])


    def set_free(self, bool_like):
        '''sets value of free parameter
        
        Parameters
        ----------
        bool_like : boolean or array of booleans
            True to free parameter
        '''

        bool_like = np.resize(bool_like, self.scalars.size)

        for i,param in enumerate(self.scalars.flatten()):
            param.set_free(bool_like[i])

    def update(self, values, source='external'):

        values = np.resize(values, self.scalars.size)

        for param, val in zip(self.scalars.flatten(), values):
            param.update(val, source)

    def summary(self):
        bounds = np.resize([str(p.bounds) for p in self.scalars.flat], self.scalars.shape)
        free = np.resize([p.free for p in self.scalars.flat] , self.scalars.shape)

        string = '{}\n{}\n'.format('Value:', self.value.__str__())
        string += '{}\n{}\n'.format('Bounds:', bounds.__str__())
        string += '{}\n{}\n'.format('Free:', free.__str__())
        print(string)

    def __getitem__(self, val):
        scalars = self.scalars[val]
        if isinstance(scalars, ParameterScalar):
            return scalars
        return ParameterArray(scalars)

    def __len__(self):
        return len(self.scalars)
        
    def __str__(self):
        return self.value.__str__()

class ParameterSpace(object):
    '''space of parameters for structural model
    
    ** Attributes **
    
    '''
    
    def __init__(self):
        self.params = OrderedDict()

    def copy(self):
        pspace = ParameterSpace()
        pspace.params = OrderedDict([(n, p.copy()) for n, p in self.params.items()])
        return pspace

    @property
    def flat(self):
        return itertools.chain(*(p.flat for p in self))

    def flatten(self, freeonly=False):
        pspace = ParameterSpace()
        if freeonly:
            pspace.params = OrderedDict([(p.name, p) for p in self.flat if p.free])
        else:
            pspace.params = OrderedDict([(p.name, p) for p in self.flat])
        return pspace

    def add_parameter(self, value, name):
        '''add a parameter to the parameter space
        
        Parameters
        ----------
        value : numeric type or array
            float, int or long, or an array of these types
        name : str
            name given to parameter
        '''
        
        if name in self.params:
            raise KeyError('Parameter with name {} already exists'.format(name))
        
        if isinstance(value, np.ndarray):
            positions = list(itertools.product(*[range(i) for i in value.shape]))
            names = ['{}[{}]'.format(name, ','.join([str(i) for i in p])) for p in positions]
            self.params.update({name:ParameterArray(np.reshape(
                [ParameterScalar(p, names[i]) for i,p in enumerate(value.flat)], value.shape))})
        elif isinstance(value, (int, long, float, np.float, np.int)):
            self.params.update({name:ParameterScalar(value, name)})
        else:
            raise ValueError('parameter value of unsupported type')
        
    
    def update(self, values, source='external'):
        '''updates free parameters with the new values
        
        Parameters
        ----------
        values: array
            1-d array of values to update the free parameters. Must
            be in order parameters were inserted.
        source : str
            'external' to update with an unconstrained value,
            'internal' to update with a constrained value
        '''

        free = self.flatten(freeonly=True)
        assert values.shape == (len(free),)
        
        for v,p in zip(values, free):
            p.update(v, source)        
                
    def __getitem__(self, name):
        return self.params[name]

    def __iter__(self):
        return self.params.itervalues()

    def __len__(self):
        return len(self.params)

    def summary(self):
        for p in self:
            p.summary()
    
    def __str__(self):
        string = []
        for name, p in self.params.items():
            string.append('{} = \n{}\n\n'.format(name, p))        
        return ''.join(string)