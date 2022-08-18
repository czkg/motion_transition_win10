import os
import numpy as np


class State():
	"""state of all the joints in a motion at one time step
	"""
	def __init__(self, q_local, ang_v, ang_a, force=None):
		"""
		Parameters:
			q_local (array) -- local quaternion of shape [22,4]
			ang_v (array) -- angular velocity of shape [22,3]
			ang_a (array) -- angular acceleration of shape [22,3]
		"""
		self.q_local = q_local
		self.ang_v = ang_v
		self.ang_a = ang_a
		self.force = force

	def compute_force_nt(self):
		"""compute force using Newton's second law (F=ma)
		Output:
			force (array) -- forces applied to all the joints, [22,3] 
		"""
		#compute bone distance r
		self.r = np.linalg.norm(x_local, axis=-1)
		#linear acceleration a = r * alpha, where alpha is angular acceleration
		self.lin_a = np.transpose(np.multiply(self.r, np.transpose(self.ang_a)))

		
		#pseudo mass
		self.m = np.ones_like(self.parents)
		#compute force
		force = np.transpose(np.transpose(self.lin_a) * self.m)
		return force

	def compute_force_dmp():
		"""compute force using DMP
		Output:
			force (array) -- forces applied to all the joints, [22, 3]
		"""
