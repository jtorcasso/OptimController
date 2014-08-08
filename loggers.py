import pandas as pd
from numpy import inf
import time

def timeit(expr):
	'''timer to evaluation expression'''
	start = time.time()
	val = expr()
	return val, time.time() - start

class NullLogger(object):
	'''base logger'''

	def __init__(self, args, parameters):
		self.args = args
		self.parameters = parameters
		self.data = self.initialize_storage()

	def initialize_storage(self):
		pass

	def store(self, fval, seconds):
		print 'Value: {} Time: {}'.format(fval, seconds)

	def __call__(self, func):

		fval, seconds = timeit(func)

		self.store(fval, seconds)

		return fval

class MemoryLogger(NullLogger):
	'''memory-based logger'''

	def __init__(self, args, parameters):
		self.args = args
		self.parameters = parameters
		self.data = self.initialize_storage()

	def initialize_storage(self):
		return pd.DataFrame(
			columns=['method', 'tol', 'fval', 'seconds'] + \
			[p.name for p in self.parameters.flatten()])

	def store(self, fval, seconds):
		'''store information from function evaluation'''

		fval_ = inf if len(self.data) == 0 else self.data['fval'].iloc[-1]

		if fval < fval_:
			values = [self.args['method'], self.args['tol'], fval, seconds] + \
				[p.value for p in self.parameters.flatten()]
			rslt = dict(zip(self.data.columns, values))
			self.data = self.data.append(rslt, ignore_index=True)

class DiskLogger(NullLogger):
	'''disk-based logger'''

	pass