'''containers for parameters used in optimization'''

from __future__ import print_function, division

# standard library
import itertools, re
from collections import OrderedDict

# third party
import numpy as np
from pandas import DataFrame

class ParameterBase(object):
    '''Base parameter class
    
    ** Attributes **
    
    '''
    
    def __init__(self, value):
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
    
    def __init__(self, value, free=True, bounds=(-np.inf, np.inf)):
        ParameterBase.__init__(self, value)
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
        
        string = '{:<10}{}\n'.format('Value:', self.value)
        string += '{:<10}{}\n'.format('Bounds:', self.bounds)
        string += '{:<10}{}\n'.format('Free:', self.free)
        print(string)

class ParameterArray(object):
    '''class to contain array of parameters'''
    
    def __init__(self, scalars):
        self.scalars = scalars
        self.shape = scalars.shape

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

    @property
    def free(self):
        return np.resize([p.free for p in self.scalars.flat] , self.scalars.shape)

    @property
    def bounds(self):
        return np.resize([str(p.bounds) for p in self.scalars.flat], self.scalars.shape)

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

        string = '{}\n{}\n'.format('Value:', self.value.__str__())
        string += '{}\n{}\n'.format('Bounds:', self.bounds.__str__())
        string += '{}\n{}\n'.format('Free:', self.free.__str__())
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

        for name,p in self.params.items():
            if hasattr(p, 'shape'):
                positions = list(itertools.product(*[range(i) for i in p.shape]))
                for pos in positions:
                    if p[pos].free | (not freeonly):
                        pspace.params.update({tuple(name) + pos:p[pos]})
            elif p.free | (not freeonly):
                pspace.params.update({tuple(name):p})

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
        search = re.compile(r'[^a-zA-Z0-9.]').search

        if search(name):
            raise ValueError('name must contain only letters a-z and digits 0-9')
        if name in self.params:
            raise KeyError('Parameter with name {} already exists'.format(name))
        
        if isinstance(value, np.ndarray):
            positions = list(itertools.product(*[range(i) for i in value.shape]))
            self.params.update({name:ParameterArray(np.reshape(
                [ParameterScalar(p) for p in value.flat], value.shape))})
        elif isinstance(value, (int, long, float, np.float, np.int)):
            self.params.update({name:ParameterScalar(value)})
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

    def to_csv(self, filepath, freeonly=False):
        '''dump parameters to csv file'''

        flat = self.flatten(freeonly=freeonly)
        index = [name for name in flat.params.keys()]
        data = [[p.value, p.free, p.bounds[0], p.bounds[1]] for p in flat]

        df = DataFrame(data, index=index, columns=['value', 'free', 'min', 'max'])
        df.to_csv(filepath, index_label='id')
    
    def from_csv(self, filepath):
        '''load parameters from csv file'''

        data = DataFrame.from_csv(filepath)
        ids = [[r.strip() for r in row[0][1:-1].split(',') if r!=''] for row in data.iterrows()]
        ids = [[v[1:-1] if i==0 else int(v) for i,v in enumerate(id_)] for id_ in ids]
        blocks = OrderedDict({ids[0][0]:[ids[0]]})
        for id_ in ids[1:]:
            if id_[0] in blocks:
                blocks[id_[0]].append(id_)
            else:
                blocks[id_[0]] = [id_]

        for name, ids in blocks.items():

            if len(ids) == 1:
                value, free, min_, max_ = data.loc[str(tuple(ids[0]))]
                self.params.update({name:ParameterScalar(value, free, (min_, max_))})

            else:
                shape = tuple([max([id_[j] for id_ in ids])+1 for j in xrange(1, len(ids[0]))])
                params = np.zeros(shape).astype('object')
                for id_ in ids:
                    value, free, min_, max_ = data.loc[str(tuple(id_))]
                    params[tuple(id_[1:])] = (ParameterScalar(value, free, (min_, max_)))

                for pos in itertools.product(*[range(i) for i in shape]):
                    if not isinstance(params[pos], ParameterScalar):
                        raise ValueError('parameter declaration incomplete at loc {}'.format(pos))

                self.params.update({name:ParameterArray(params)})

    def __getitem__(self, name):
        return self.params[name]

    def __iter__(self):
        return self.params.itervalues()

    def __len__(self):
        return len(self.params)

    def summary(self):
        for name, p in self.params.items():
            print("{} = \n".format(name))
            p.summary()
    
    def __str__(self):
        string = []
        for name, p in self.params.items():
            string.append('{} = \n{}\n\n'.format(name, p))        
        return ''.join(string)