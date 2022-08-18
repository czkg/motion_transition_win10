import os
from data.base_dataset import BaseDataset
import scipy.io
import torch
import numpy as np
from glob import glob
import pickle
import math


class LafanDataset(BaseDataset):
	""" This dataset class can load Lafan data specified by the file path --dataroot/path/to/data
	"""

	def __init__(self, opt):
		""" Initialize this dataset class

		Parameters:
			opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""

		BaseDataset.__init__(self, opt)
		self.name2accnum = {}
		self.name2seqaccnum = {}
		self.mode = opt.lafan_mode
		self.window = opt.lafan_window
		self.offset = opt.lafan_offset
		self.samplerate = opt.lafan_samplerate

		files = glob(os.path.join(self.root, '*.pkl'))

		self.pose_count = 0
		self.seq_count = 0
		for i, f in enumerate(files):
			with open(f, 'rb') as ff:
				rawdata = pickle.load(ff, encoding='latin1')
			q_local = rawdata['q_local']
			force = rawdata['force']
			data = q_local
			self.name2accnum[files[i]] = data.shape[0] + self.pose_count
			# math.floor((L-W)/off) + 1
			self.name2seqaccnum[files[i]] = math.floor((data[::self.samplerate].shape[0] - self.window)/self.offset) + 1 + self.seq_count 
			self.pose_count += data.shape[0]
			self.seq_count += math.floor((data[::self.samplerate].shape[0] - self.window)/self.offset) + 1
		self.accnum2names = dict((v,k) for k,v in self.name2accnum.items())
		self.seqaccnum2names = dict((v,k) for k,v in self.name2seqaccnum.items())


	def generate_filename(self, base_name, start, end):
		"""Generate new file name given parent name, start idx and end idx

		Parameters:
			base_name -- parent name from originl file
			start     -- start idx in base_name file
			end       -- end idx in base_name file
		"""

		new_name = base_name[:-4] + '_' + str(start) + '_' + str(end) + '.pkl'
		return new_name


	def __getitem__(self, index):
		""" Return a data point and its metadata information

		Parameters:
			index -- a random integer for data indexing

		Returns a dictionary that contains data and path
		"""
		if self.mode == 'pose':
			upper = [k for k in self.accnum2names.keys() if index < k]
			lower = [k for k in self.accnum2names.keys() if index >= k]
			k = upper[0]
			if len(lower) > 0:
				k_prev = lower[-1]
				in_idx = index - k_prev
			else:
				in_idx = index
			file = self.accnum2names[k]
			with open(file, 'rb') as f:
				rawdata = pickle.load(f, encoding='latin1')
			q_local = rawdata['q_local']
			force = rawdata['force']
			pose = q_local[in_idx]

			pose = pose.reshape(-1)

			return torch.tensor(pose)
		elif self.mode == 'seq':
			upper = [k for k in self.seqaccnum2names.keys() if index < k]
			lower = [k for k in self.seqaccnum2names.keys() if index >= k]
			k = upper[0]
			if len(lower) > 0:
				k_prev = lower[-1]
				init_idx = index - k_prev
			else:
				init_idx = index
			file = self.seqaccnum2names[k]
			with open(file, 'rb') as f:
				rawdata = pickle.load(f, encoding='latin1')
			q_local = rawdata['q_local']
			force = rawdata['force']
			lin_a = rawdata['lin_a']

			q_local = q_local[::self.samplerate][init_idx*self.offset : init_idx*self.offset+self.window]
			q_local = np.asarray(q_local)
			force = force[::self.samplerate][init_idx*self.offset : init_idx*self.offset+self.window]
			force = np.asarray(force)
			lin_a = lin_a[::self.samplerate][init_idx*self.offset : init_idx*self.offset+self.window]
			lin_a = np.asarray(lin_a)

			#generate new file name
			file_name = self.generate_filename(file.split('/')[-1], init_idx*self.offset, init_idx*self.offset+self.window)

			# add rv
			# if self.is_local is True:
			# 	global_file = file[:-9] + 'global.pkl'
			# else:
			# 	global_file = file
			# with open(os.path.join(self.root, global_file), 'rb') as f:
			# 	global_data = pickle.load(f, encoding='latin1')
			# rv = global_data['rv'][::self.framerate][init_idx*self.offset : init_idx*self.offset+self.window]
			# rv = rv[:,np.newaxis,...]
			# seq = np.concatenate((rv, seq), axis=1)

			q_local = q_local.reshape(q_local.shape[0], -1)
			force = force.reshape(force.shape[0], -1)
			lin_a = lin_a.reshape(lin_a.shape[0], -1)

			return {'q_local': torch.tensor(q_local), 'force': torch.tensor(force), 'lin_a': torch.tensor(lin_a), 'info': file_name}
		else:
			raise('Invalid mode!')

	def __len__(self):
		if self.mode == 'pose':
			return self.pose_count
		else:
			return self.seq_count
		