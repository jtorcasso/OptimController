import pandas as pd
from numpy import inf, dtype
import time
import datetime

try:
	import tables
	import_tables = True
except:
	import_tables = False

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
		NullLogger.__init__(self, args, parameters)

	def initialize_storage(self):
		return pd.DataFrame(
			columns=['method', 'tol', 'fval', 'seconds'] + \
			[str(n) for n in self.parameters.flatten().params])

	def store(self, fval, seconds):
		'''store information from function evaluation to memory'''

		fval_ = inf if len(self.data) == 0 else self.data['fval'].iloc[-1]

		if fval < fval_:
			values = [self.args['method'], self.args['tol'], fval, seconds] + \
				[p.value for p in self.parameters.flatten()]
			rslt = dict(zip(self.data.columns, values))
			self.data = self.data.append(rslt, ignore_index=True)

class DiskLogger(NullLogger):
	'''disk-based logger'''

	def __init__(self, args, parameters, **kwargs):

		if not import_tables:
			raise ImportError('could not import "tables"')

		self.filepath = kwargs.get('filepath', 'logging.h5')
		self.groupname = kwargs.get('groupname', '{}'.format(datetime.date.today()))
		now = datetime.datetime.now()
		self.tablename = kwargs.get(
			'tablename', 'log-{}:{}:{}'.format(now.hour, now.minute, now.second))

		NullLogger.__init__(self, args, parameters)

	def initialize_storage(self):

		hfile = tables.open_file(self.filepath, 'a')

		dt = dtype([('method', 'S40'), ('tol', float), ('fval', float),
			('seconds', float)] + [(str(n), float) for n in self.parameters.flatten().params])

		return hfile.createTable('/{}'.format(self.groupname), self.tablename, dt, 
			createparents=True)

	def store(self, fval, seconds):
		'''store information from function evaluation to disk'''

		fval_ = inf if len(self.data) == 0 else self.data.cols.fval[:][-1]

		if fval < fval_:
			values = [self.args['method'], self.args['tol'], fval, seconds] + \
				[p.value for p in self.parameters.flatten()]
			self.data.append([tuple(values)])
			self.data.flush()

	def read(self):
		'''read table into memory (in a pandas dataframe)'''

		return pd.DataFrame.from_records(self.data[:])

	def close(self):
		'''closes the logging file'''

		self.data.close()