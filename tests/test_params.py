import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from parameters import ParameterSpace
from scipy.optimize import fmin_bfgs
import numpy as np


np.random.seed(1234)

p = ParameterSpace()
p.add_parameter(0, 'b')
p['b'].set_bounds(-10, 10000)

x = np.random.randn(1000)

y = 2*x + np.random.randn(1000)

def squares(params,p,y,x):

    print(params)
    p.update(params)
    print(p)
    return np.square(y - p['b'].value*x).sum()

init = p['b'].value_

fmin = fmin_bfgs(squares, x0=init, args=(p,y,x))

p.summary()