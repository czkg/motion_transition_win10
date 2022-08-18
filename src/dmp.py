import os
import numpy as np
import utils.utils
import math


class dmpQ():
	""" This class implement discrete quaternion DMP:

	"""
	def __init__(self, goal, r, numInBetween):
		"""
		Paramters:
			init (state) -- init state including quat, angular vel and angular acc
			goal (state) -- goal state including quat, angular vel and angular acc
			r (vec) -- distance to parents for all the joints
			numInBetween (int) -- number of frames in between
		"""
		# tau^2 / K = mr (m is the mass of joint, and r is the distance to its parent)
		# here we set tau = 1, m = 1, then K = 1/r 
		self.alpha = 1
		self.tau = 1
		self.K = 1. / r
		self.d = 2 * math.sqrt(self.K * self.tau)
		# 30 fps
		self.dt = 1./30.
		# number of dt in each DMP time duration
		self.ndt = 1
		self.clk = 1.
		self.goal = goal

		#moving target at time 0 (s = 1)
		T = self.dt * numInBetween
		self.mtZero = utils.quatProduct(utils.quatExp(T/-2.*self.goal.ang_v), self.goal.q_local)

	def compute_next_state(self, quat, force, ang_v):
		"""compute next state given current state including quat, ang_v, ang_a and force
		Parameters:
			quat: current state quaterion [n_joints, 4]
			force: current state force [n_joints, 3]
			ang_v: current state angular velocity [n_joints, 3]
		"""
		self.clk = self.clk + (-self.alpha*self.clk)*self.dt/self.tau
		mt = utils.quatProduct(utils.quatExp(self.tau*np.log(self.clk)/(-2.*self.alpha)*self.goal.ang_v), self.mtZero)
		quatErr = utils.quatError(mt, quat)

		next_ang_a = (self.K * ((1 - self.clk) * quatErr + force) + 
				self.d * (self.goal.ang_v - ang_v) * (1 - self.clk)) / self.tau
		next_ang_v = ang_v + next_ang_a * self.dt
		next_quat = utils.quatIntegral(quat, next_ang_v, self.dt)

		next_state = State(next_quat, next_ang_v, next_ang_a)
		return next_state




class dmpP():
	""" This class implement discrete quaternion DMP:

	"""
	def __init__(self, goal, r, numInBetween):
		"""
		Paramters:
			init (state) -- init state including quat, angular vel and angular acc
			goal (state) -- goal state including quat, angular vel and angular acc
			r (vec) -- distance to parents for all the joints
			numInBetween (int) -- number of frames in between
		"""
		# tau^2 / K = mr (m is the mass of joint, and r is the distance to its parent)
		# here we set tau = 1, m = 1, then K = 1/r
		self.alpha = 1
		self.tau = 1
		self.K = 1. / r
		self.d = 2 * math.sqrt(self.K * self.tau)
		# 30 fps
		self.dt = 1./30.
		# number of dt in each DMP time duration
		self.ndt = 1
		self.clk = 1.
		self.goal = goal

		#moving target at time 0 (s = 1)
		T = self.dt * numInBetween
		self.mtZero = utils.quatProduct(utils.quatExp(T/-2.*self.goal.ang_v), self.goal.q_local)

	def compute_next_state(self, quat, force, ang_v):
		"""compute next state given current state including quat, ang_v, ang_a and force
		Parameters:
			quat: current state quaterion [n_joints, 4]
			force: current state force [n_joints, 3]
			ang_v: current state angular velocity [n_joints, 3]
		"""
		self.clk = self.clk + (-self.alpha*self.clk)*self.dt/self.tau
		mt = utils.quatProduct(utils.quatExp(self.tau*np.log(self.clk)/(-2.*self.alpha)*self.goal.ang_v), self.mtZero)
		quatErr = utils.quatError(mt, quat)

		next_ang_a = (self.K * ((1 - self.clk) * quatErr + force) +
				self.d * (self.goal.ang_v - ang_v) * (1 - self.clk)) / self.tau
		next_ang_v = ang_v + next_ang_a * self.dt
		next_quat = utils.quatIntegral(quat, next_ang_v, self.dt)

		next_state = State(next_quat, next_ang_v, next_ang_a)
		return next_state
        	
        