import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation

eps = 1e-5

def slerp(p0, p1, t):
    """
    Spherical linear interpolation
    """
    val = np.dot(np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1)))
    val = val if val >= -1 else -1
    val = val if val < 1 else 1
    omega = np.arccos(val)
    so = np.sin(omega)
    if so == 0.:
        so += eps
    return np.sin(1.0 - t) * omega / so * p0 + np.sin(t * omega) / so * p1

def blend(outputs, target):
        """Calculate interpolation between choosen outputs and target
        Parameters:
            outputs (array) -- array of latent codes [len_sequence, z_dim]
            target (array) -- array of latent codes [z_dim]
        Returns:
            transition (array) -- transition frames [len_sequence, z_dim]
        """
        # find output with least L2 distance to target
        o2t = outputs - target
        o2o = outputs[1:] - outputs[:-1]
        o2t = np.linalg.norm(o2t, axis=-1)
        o2o = np.linalg.norm(o2o, axis=-1)
        o2o_max = np.amax(o2o, axis=0)

        o2t_min = np.amin(o2t, axis=0)
        o2t_minidx = np.argmin(o2t, axis=0)

        need_blend = o2t_min > o2o_max
        transition = outputs[:o2t_minidx]
        if need_blend:
            n_steps = int(np.ceil(o2t_min / o2o_max))
            blendings = np.array([slerp(outputs[o2t_minidx], target, t) for t in np.linspace(0, 1, n_steps)])
            transition = np.concatenate((transition, blendings), axis=0)

        return transition

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def quatElemProduct(q1, q2):
	"""Parameters:
		q1: quaternion of shape [4,]
		q2: quaternion of shape [4,]
	"""
	real = q1[0]*q2[0] - np.matmul(q1[np.newaxis, 1:], q2[1:, np.newaxis])[0][0]
	imag = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:],q2[1:])

	res = np.zeros_like(q1)
	res[0] = real
	res[1:] = imag
	return res

def quatProduct(q1, q2):
	"""Parameters:
		q1: quaternion of shape [22, 4]
		q2: quaternion of shape [22, 4]
	   Output:
	   	res: products of shape [22, 4]
	"""
	if q1.shape[-1]!=4 or q2.shape[-1]!=4:
		raise Exception('quaternion must has dimension 4')

	res = np.asarray([quatElemProduct(q1[i],q2[i]) for i in range(q1.shape[0])])
	return res


def quatElemExp(r):
	"""Parameters:
		r: vector of shape [3,]
	"""
	nR = np.linalg.norm(r)
	if nR > np.pi:
		raise Exception('||r|| > pi')

	q = np.zeros(r.shape+1)

	if nR > 1e-8:
		q[0] = np.cos(nR)
		q[1:] = np.sin(nR)*r/nR
	else:
		q[0] = 1.
	return q


def quatExp(r):
	"""Parameters:
		r: vector of shape [22, 3]
	   Output: 
	   	q: quaternion of shape [22, 4]
	"""
	if r.shape[-1]!=3:
		raise Exception('vector must has diemnsion 3')

	q = np.asarray([quatElemExp(r[i]) for i in range(r.shape[0])])
	return q


def quatElemConjugate(q):
	"""Parameters:
		q: quaternion of shape [4,]
	"""
	q[1] = -q[1]
	q[2] = -q[2]
	q[3] = -q[3]
	return q

def quatConjugate(q):
	"""Parameters:
		q: quaternion of shape [22, 4]
	"""
	res = np.asarray([quatElemConjugate(q[i]) for i in range(q.shape[0])])
	return res


def quatErrorVec(q1, q2):
	"""Parameters:
		q1: quaternion of shape [22, 4]
		q2: quaternion of shape [22, 4]
	   Output:
	   	err: errors of shape [22, 4]
	"""
	err = 2*quatProduct(q1, quatConjugate(q2))[:,1:]


# def quatErrorLog(q1, q2):
# 	"""To do
# 	"""

def quatError(q1, q2, method='vec'):
	"""Parameters:
		q1: quaternion of shape [22, 4]
		q2: quaternion of shape [22, 4]
		method: vec or log
	   Output:
	   	err: errors of shape [22, 4]
	"""
	if q1.shape[-1]!=4 or q2.shape[-1]!=4:
		raise Exception('quaternion must has dimension 4')

	if method == 'vec':
		err = quatErrorVec(q1, q2)
	elif method == 'log':
		err = quatErrorLog(q1, q2)
	else:
		raise Exception('Unknown quatError method.')

	return err


def quatIntegral(quat, ang_v, dt):
	"""Returns q(t+dt) by numerically integrating the angular velocity 
	Parameters:
		quat: quaternion of shape [22, 4]
		ang_v: angular velocity of shape [22, 3]
		dt: differential of time
	"""
	if quat.shape[-1]!=4 or ang_v.shape[-1]!=3:
		raise Exception('quaternion must has dimension 4 and angular velocity must has dimension 3')
	exp = quatExp(dt*ang_v/2.)
	qNext = quatProduct(exp, quat)
	return qNext



def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(quat_mul_vec(gr[parents[i]], lpos[..., i:i+1, :]) + gp[parents[i]])
        gr.append(quat_mul    (gr[parents[i]], lrot[..., i:i+1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res


def norm_x(x):
	"""
	Performs nomalization for global position data x

	"param x: global position data of shape [joint, 3]
	"""
	# move root to center [0,0,0]
	x = x - x[0]

	# find largest joint
	dist = np.amax(abs(x))
	x = x / (dist*1.2)

	return x


def quat2euler(q):
	"""
	transform quaternion to euler angle
	param q: quaternion of shape [len, joint, 4]
	"""
	rot = np.zeros_like(q)[...,:3]
	for i in range(q.shape[0]):
		for j in range(q.shape[1]):
			rot[i][j] = quaternion_to_euler_angle_vectorized(q[i][j][0], q[i][j][1], q[i][j][2], q[i][j][3])
	return rot

def quaternion_to_euler_angle_vectorized(w, x, y, z):
	ysqr = y * y

	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = np.degrees(np.arctan2(t0, t1))

	t2 = +2.0 * (w * y - z * x)

	t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
	Y = np.degrees(np.arcsin(t2))

	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = np.degrees(np.arctan2(t3, t4))

	return [X, Y, Z]