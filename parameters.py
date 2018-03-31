'''containers for parameters used in optimization

TO DO
-----
1. keep update for only the parameter space?
2. write class for symmetric array
3. write class for spd array

note: might be better to keep to_external and to_internal as arrays of functions
'''

from __future__ import print_function, division

# standard library
import itertools, re
from collections import OrderedDict

# third party
import numpy as np
from pandas import DataFrame, read_csv

def symmetric_to_upper(X):
    '''convert matrix to a flattened array of elements in upper triangle
    
    Parameters
    ----------
    X : array
        symmetric matrix
    Returns
    -------
    x : array
        1-d array of upper triangular elements
    '''
    x = X[np.triu_indices(X.shape[0])]
    return x

def upper_to_symmetric(x):
    '''convert upper triangular elements to symmetric matrix
    
    Parameters
    ----------
    x : array
        1-d array of upper triangular elements
    Returns
    -------
    X : array
        symmetric matrix
    '''
    elements = len(x)
    size = int((-1 + np.sqrt(1 + 8*elements))/2)
    X = np.empty((size, size), dtype=x.dtype)
    indices = np.triu_indices(size)
    X[indices] = x
    X[(indices[1], indices[0])] = x
    return X

def is_symmetric(X):
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2

    if X.shape[0] == X.shape[1]:
        if (X == X.T).all():
            return True
    return False

def get_transforms(bounds):

    min_, max_ = bounds
    if (not np.isfinite(min_)) & (not np.isfinite(max_)):
        to_internal = lambda val: val
        to_external = lambda val: val
    elif np.isfinite(min_):
        to_external = lambda val: np.sqrt((val - min_ + 1)**2 - 1)
        to_internal = lambda val: min_ - 1 + np.sqrt(val**2 + 1)
    elif np.isfinite(max_):
        to_external = lambda val: np.sqrt((max_ - val + 1)**2 - 1)
        to_internal = lambda val: max_ + 1 - np.sqrt(val**2 + 1)
    else:
        to_external = lambda val: np.arcsin(2*(val - min_)/(max_ - min_) - 1)
        to_internal = lambda val: min_ + (np.sin(val) + 1)*(max_ - min_)/2

    return (to_external, to_internal)

def ParameterScalar(object):

    def __init__(self, value):
        self.value = value
        self.value_ = value
        self.bounds = (-np.inf, np.inf)
        self.free = True
        self.to_external = lambda val: val
        self.to_internal = lambda val: val

    def _update_transform(self):
        self.to_external, self.to_internal = get_transforms(self.bounds)

        self.value_ = self.to_external(self.value)

    def set_bounds(self, bounds):
        self.bounds = bounds

    def set_free(self, free):
        self.free = free

    def update(self, value):
        '''updates value of parameter from external, unconstrained, space; only
        for the free parameters
        
        Parameters
        ----------
        value : numeric type
            value to update with
        source : str
            'external' to update with an unconstrained value, i.e., transforms
            input to value consistent with the model
            'internal' to update with a constrained value
        '''
        assert isinstance(value, (int, float, np.float, np.int))
        assert self.free

        self.value_ = value
        self.value = self.to_internal(self.value_)

    def summary(self):
        '''print summary of the parameter'''
        
        string = '{:<10}{}\n'.format('Value:', self.value)
        string += '{:<10}{}\n'.format('Bounds:', self.bounds)
        string += '{:<10}{}\n'.format('Free:', self.free)
        print(string)

    def __str__(self):
        return self.value.__str__()

    def __repr__(self):
        return self.value.__repr__()

class ParameterSlice(object):

    def __init__(self, ix, parray):
        self.parray = parray
        self.ix = ix

    def set_bounds(self, bounds):
        self.parray.bounds[0][self.ix] = bounds[0]
        self.parray.bounds[1][self.ix] = bounds[1]

        self.parray._update_transform()

    def set_free(self, free):
        self.parray.free[self.ix] = free

class ParameterArray(object):

    def __init__(self, value):
        self.value = value
        self.value_ = value.flatten()
        self.shape = value.shape
        self.size = value.size
        self.bounds = (np.resize(-np.inf, self.value_.shape), np.resize(np.inf, self.value_.shape))
        self.free = np.resize(True, self.value_.shape)
        self.to_external = lambda val: val.flatten()
        self.to_internal = lambda val: np.resize(val, self.shape)

    def _update_transform(self):
        transforms = [get_transforms((l, u)) for l, u in zip(self.bounds[0].flat, self.bounds[1].flat)]
        self.to_external = lambda val: np.array([t[0](v) for t,v in zip(transforms, val.flat)])
        self.to_internal = lambda val: np.resize([t[1](v) for t,v in zip(transforms, val.flat)], self.shape)

        self.value_ = self.to_external(self.value)

    def set_bounds(self, bounds):
        '''set bounds for the values
        '''
        self.bounds[0][:] = bounds[0]
        self.bounds[1][:] = bounds[1]

        self._update_transform()

    def set_free(self, free):
        '''set parameter value to free or fixed'''
        self.free[:] = free

    def update(self, value):
        '''updates value of parameter from external, unconstrained, space; only
        for the free parameters
        
        Parameters
        ----------
        value : numeric type
            value to update with
        source : str
            'external' to update with an unconstrained value, i.e., transforms
            input to value consistent with the model
            'internal' to update with a constrained value
        '''
        assert value.shape == self.value_[self.free].shape

        self.value_[self.free] = value
        self.value = self.to_internal(self.value_)

    def summary(self):

        string = '{}\n{}\n'.format('Value:', self.value.__str__())
        bounds = np.reshape([str((a,b)) for a,b in zip(self.bounds[0].flat, self.bounds[1].flat)], self.shape)
        string += '{}\n{}\n'.format('Bounds:', bounds.__str__())
        free = np.resize(self.free, self.shape)
        string += '{}\n{}\n'.format('Free:', free.__str__())
        print(string)

    def __getitem__(self, ix):

        ix = list(np.reshape(np.arange(self.size), self.shape)[ix].flatten())

        return ParameterSlice(ix, self)

    def __len__(self):
        return self.shape[0]
        
    def __str__(self):
        return self.value.__str__()

class ParameterSymmetricArray(object):

    def __init__(self, value):
        self.value = value
        self.value_ = symmetric_to_upper(value)
        self.shape = value.shape
        self.size = value.size
        self.bounds = (np.resize(-np.inf, self.value_.shape), 
                       np.resize(np.inf, self.value_.shape))
        self.free = np.resize(True, self.value_.shape)
        self.to_external = lambda val: symmetric_to_upper(val)
        self.to_internal = lambda val: upper_to_symmetric(val)

    def _update_transform(self):
        transforms = [get_transforms((l, u)) for l, u in zip(self.bounds[0].flat, self.bounds[1].flat)]
        self.to_external = lambda val: np.array([t[0](v) for t,v in zip(transforms, symmetric_to_upper(val))])
        self.to_internal = lambda val: upper_to_symmetric(np.array([t[1](v) for t,v in zip(transforms, val.flat)]))

        self.value_ = self.to_external(self.value)

    def set_bounds(self, bounds):
        self.bounds[0][:] = bounds[0]
        self.bounds[1][:] = bounds[1]

        self._update_transform()

    def set_free(self, free):
        self.free[:] = free

    def update(self, value):
        '''updates value of parameter from external, unconstrained, space; only
        for the free parameters
        
        Parameters
        ----------
        value : numeric type
            value to update with
        source : str
            'external' to update with an unconstrained value, i.e., transforms
            input to value consistent with the model
            'internal' to update with a constrained value
        '''
        assert value.shape == self.value_[self.free].shape

        self.value_[self.free] = value
        self.value = self.to_internal(self.value_)

    def summary(self):

        string = '{}\n{}\n'.format('Value:', self.value.__str__())
        b_low = upper_to_symmetric(self.bounds[0]).flat
        b_high = upper_to_symmetric(self.bounds[1]).flat
        bounds = np.reshape([str((a,b)) for a,b in zip(b_low, b_high)], self.shape)
        string += '{}\n{}\n'.format('Bounds:', bounds.__str__())
        string += '{}\n{}\n'.format('Free:', upper_to_symmetric(self.free).__str__())
        string += '{} {}\n'.format('Constraint:', 'Symmetric')

        print(string)

    def __getitem__(self, ix):

        ix = list(np.unique(upper_to_symmetric(np.arange(len(self.value_)))[ix].flatten()))

        return ParameterSlice(ix, self)

    def __len__(self):
        return self.shape[0]
        
    def __str__(self):
        return self.value.__str__()

class ParameterSPDArray(object):

    def __init__(self, value):
        self.value = value
        self.value_ = np.linalg.cholesky(value)[np.tril_indices(value.shape[0])]
        self.shape = value.shape
        self.size = value.size
        self.free = np.resize(True, self.value_.shape)
        self.to_external = lambda val: np.linalg.cholesky(val)[np.tril_indices(self.shape[0])]

    def to_internal(self, val):
        chol = np.zeros(self.shape)
        chol[np.tril_indices(self.shape[0])] = val
        return chol.dot(chol.T)

    def set_free(self, free):
        self.free[:] = free

    def update(self, value):
        '''updates value of parameter from external, unconstrained, space; only
        for the free parameters
        
        Parameters
        ----------
        value : numeric type
            value to update with
        source : str
            'external' to update with an unconstrained value, i.e., transforms
            input to value consistent with the model
            'internal' to update with a constrained value
        '''
        assert value.shape == self.value_[self.free].shape

        self.value_[self.free] = value
        self.value = self.to_internal(self.value_)

    def summary(self):

        string = '{}\n{}\n'.format('Value:', self.value.__str__())
        string += '{} {}\n'.format('Bounds:', 'NA')
        string += '{} {}\n'.format('Free:', self.free[0])
        string += '{} {}\n'.format('Constraint:', 'Symmetric Positive Definite')
        print(string)

    def __len__(self):
        return self.shape[0]
        
    def __str__(self):
        return self.value.__str__()



class ParameterSpace(object):
    '''space of parameters for structural model
    
    ** Attributes **
    
    '''
    
    def __init__(self):
        self.params = OrderedDict()
        self.ptypes = []

    def add_parameter(self, value, name, ptype='infer'):
        '''add a parameter to the parameter space
        
        Parameters
        ----------
        value : numeric type or array
            float, int or long, or an array of these types
        name : str
            name given to parameter
        ptype : str
            "infer" to infer ptype, "scalar", "array", "symmetric", "spd" for symmetric positive definite
        '''
        assert isinstance(value, (int, float, np.float, np.int, np.ndarray))
        if isinstance(value, np.ndarray):
            assert (value.ndim <= 2) & (value.size > 1)

        if ptype == "infer":
            if isinstance(value, (int, float, np.float, np.int)):
                ptype = "scalar"
            else:
                ptype = "array"
        elif ptype == "scalar":
            assert isinstance(value, (int, float, np.float, np.int))
        elif ptype == "array":
            assert isinstance(value, np.ndarray)
        elif ptype == "symmetric":
            assert is_symmetric(value)
        elif ptype == "spd":
            try:
                np.linalg.cholesky(value)
            except:
                raise ValueError("value is not consistent with spd ptype")
        else:
            raise ValueError("invalid ptype")

        search = re.compile(r'[^a-zA-Z0-9.]').search

        if search(name):
            raise ValueError('name must contain only letters a-z and digits 0-9')
        if name in self.params:
            raise KeyError('Parameter with name {} already exists'.format(name))
        
        if ptype == "scalar":
            self.params.update({name:ParameterScalar(value)})
        elif ptype == "array":
            self.params.update({name:ParameterArray(value)})
        elif ptype == "symmetric":
            self.params.update({name:ParameterSymmetricArray(value)})
        else:
            self.params.update({name:ParameterSPDArray(value)})

        self.ptypes.append(ptype)

    @property
    def value_(self):
        vals = []
        for p, ptype in zip(self.params.values(), self.ptypes):
            if ptype == "scalar":
                if p.free:
                    vals.append(p.value_)
            else:
                vals.append(p.value_[p.free])
        return np.hstack(vals)


    def update(self, values):
        '''updates free parameters with the new values
        
        Parameters
        ----------
        values: array
            1-d array of values to update the free parameters. Must
            be in order parameters were inserted.
        '''

        assert values.shape == self.value_.shape
        
        i = 0
        for p, ptype in zip(self.params.values(), self.ptypes):
            if ptype == "scalar":
                if p.free:
                    p.update(values[i])
                    i += 1
            else:
                num_free = len(p.value_[p.free])
                p.update(values[i:i+num_free])
                i += num_free

    def to_dict(self):

        params = OrderedDict()

        for (name,p),ptype in zip(self.params.items(), self.ptypes):
            if ptype == "scalar":
                params.update({name:p.value})
            else:
                positions = list(itertools.product(*[range(i) for i in p.shape]))
                for pos, v in zip(positions, p.value.flat):
                    params.update({(name,) + pos:v})

        return params


    def to_csv(self, filepath):
        '''dump parameters to csv file'''

        rows = [['pname', 'ptype', 'pshape', 'value', 'free', 'min', 'max']]
        for (name, p), ptype in zip(self.params.items(), self.ptypes):
            if ptype == "scalar":
                rows = np.vstack([rows, [[name, ptype, 'NA', p.value, p.free, p.bounds[0], p.bounds[1]]]])
            else: 
                N = p.value.size
                if ptype == "array":
                    free = p.free
                    min_, max_ = p.bounds[0], p.bounds[1]
                elif ptype == "spd":
                    free = [p.free[0]]*N
                    min_, max_ = ['NA']*N, ['NA']*N
                else:
                    free = upper_to_symmetric(p.free).flatten()
                    min_ = upper_to_symmetric(p.bounds[0]).flatten()
                    max_ = upper_to_symmetric(p.bounds[1]).flatten()

                row = np.vstack([[name]*N, [ptype]*N, [str(p.shape)]*N, p.value.flatten(), free, min_, max_]).T
                rows = np.vstack([rows, row])
        df = DataFrame(rows)
        df.to_csv(filepath, header=False, index=False)
    
    def from_csv(self, filepath):
        '''load parameters from csv file'''

        df = read_csv(filepath)
        pnames = df.pname.drop_duplicates().tolist()

        for pname in pnames:
            ptype = df[df.pname == pname]['ptype'].iloc[0]
            pshape = df[df.pname == pname]['pshape'].iloc[0]
            value = df[df.pname == pname]['value'].as_matrix()
            free = df[df.pname == pname]['free'].as_matrix()
            min_ = df[df.pname == pname]['min'].as_matrix()
            max_ = df[df.pname == pname]['max'].as_matrix()

            if ptype == "scalar":
                self.add_parameter(value[0], pname, ptype)
                self[pname].set_free(free[0])
                self[pname].set_bounds((min_[0], max_[0]))
            else:
                pshape = pshape[1:-1].split(',')
                if pshape[1] == '':
                    pshape = (int(pshape[0]),)
                else:
                    pshape = (int(pshape[0]), int(pshape[1]))

                value = np.resize(value, pshape)

                if ptype == "spd":
                    self.add_parameter(value, pname, ptype)
                    self[pname].set_free(free[0])
                elif ptype == "array":
                    self.add_parameter(value, pname, ptype)

                    self[pname].free = free
                    self[pname].bounds = (min_, max_)
                    self[pname]._update_transform()
                else:
                    self[pname].free = symmetric_to_upper(np.reshape(free, pshape)).flatten()
                    self[pname].bounds = (symmetric_to_upper(np.reshape(min_, pshape)).flatten(),
                                        symmetric_to_upper(np.reshape(max_, pshape)).flatten())
                    self[pname]._update_transform()
        

    def __getitem__(self, name):
        return self.params[name]

    def __iter__(self):
        return iter(self.params.values())

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