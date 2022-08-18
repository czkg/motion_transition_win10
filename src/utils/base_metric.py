import numpy as np
import os
import torch
from abc import abstractmethod
import abc
import sys

if sys.version_info >= (3,4):
	ABC = abc.ABC
else:
	ABC = abc.ABCMeta('ABC', (), {})


class BaseMetric(ABC):
	def __init__(self, opt):
		self.opt = opt
		self.gpu_ids = opt.gpu_ids
		self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')