import PySide2
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.Qt3DCore import *
from PySide2.Qt3DExtras import *
from PySide2 import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import QtGui
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import pickle
from pathlib import Path
from PIL import Image
import time
import trimesh
from OpenGL.GL import *

import torch
import torch.nn as nn

from utils.utils import *
from dmp import dmpQ, dmpP

import maya.standalone
import maya.cmds as cmds

from options.test_options import TestOptions
from models import create_model
from data import create_dataset
from utils.visualizer import Visualizer

model = "../model/model_skeleton.fbx"
mesh = "../model/model_skeleton.obj"
faces = None

parents = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11, 12, 11, 14, 15,
           16, 11, 18, 19, 20]

transition_modes = ['Number in between', 'Number of FPS']

model_joints = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'RightShoulder', 'RightArm',
			   'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3',
			   'RightHandThumb4', 'RightInHandIndex', 'RightHandIndex1', 'RightHandIndex2',
			   'RightHandIndex3', 'RightHandIndex4', 'RightInHandMiddle', 'RightHandMiddle1',
			   'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightInHandRing',
			   'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightInHandPinky',
			   'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'LeftShoulder',
			   'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3',
			   'LeftHandThumb4', 'LeftInHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
			   'LeftHandMiddle4', 'LeftInHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
			   'LeftHandRing4', 'LeftInHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',
			   'LeftHandPinky4', 'LeftInHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
			   'LeftHandIndex4', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe', 'RightToeEnd',
			   'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe', 'LeftToeEnd']

model_trunk_joints = ['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe', 'RightUpLeg', 'RightLeg',
					  'RightFoot', 'RightToe', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder',
					  'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm','RightForeArm', 'RightHand']


def mayaInit():
	maya.standalone.initialize(name='python')
	cmds.loadPlugin("objExport")
	cmds.loadPlugin("fbxmaya")

	#load maya model
	cmds.file(model, o=True)
	global faces
	faces = trimesh.load_mesh(mesh).faces


def mayaUpdateJoint(rx, ry, rz, idx):
	"""
	Update each joint
	Params:
		rx: rotation alone x axis
		ry: rotation alone y axis
		tz: rotation alone z axis
		idx: index of current joint
	"""
	attrX = 'Model:' + model_trunk_joints[idx] + '.rotateX'
	attrY = 'Model:' + model_trunk_joints[idx] + '.rotateY'
	attrZ = 'Model:' + model_trunk_joints[idx] + '.rotateZ'

	cmds.setAttr(attrX, rx)
	cmds.setAttr(attrY, ry)
	cmds.setAttr(attrZ, rz)


def getMayaUpdateMesh(rot, off):
	"""
		Update mesh in Maya
		Params:
			rot: rx, ry, rz
			off: offset from first frame
		Params:
			rx: array of rotation alone x axis
			ry: array of rotation alone y axis
			tz: array of rotation alone z axis
		"""
	rx = rot[:,0]
	ry = rot[:,1]
	rz = rot[:,2]
	for i in range(len(model_trunk_joints)):
		mayaUpdateJoint(rx[i], ry[i], rz[i], i)
	m = cmds.ls(type='mesh')[1]
	cmds.select(m)
	verts = np.asarray(cmds.getAttr(m + '.vrts[:]')) + off

	# rotate mesh
	r = R.from_euler('xyz', [90, 0, 180], degrees=True)
	rr = r.as_matrix()
	verts = np.matmul(verts, rr) / 50.
	return gl.MeshData(vertexes=verts, faces=faces)


class dmtSettingWindow(QWidget):
	procDone = QtCore.Signal(list)
	def __init__(self):
		super(dmtSettingWindow, self).__init__()
		self.initUI()

	def initUI(self):
		self.resize(600, 60)
		self.center()
		self.setWindowTitle('DMT settings')

		# with default values
		tft_model_label = QLabel('TFT Model')
		transition_mode_label = QLabel('Transition Mode')
		num_inbetween_label = QLabel('Num of frames in between')
		num_fps_label = QLabel('Number of frames per second')

		tft_model_label.setFixedHeight(50)
		transition_mode_label.setFixedHeight(50)
		num_inbetween_label.setFixedHeight(50)
		num_fps_label.setFixedHeight(50)

		tft_model_label.setFixedWidth(360)
		transition_mode_label.setFixedWidth(360)
		num_inbetween_label.setFixedWidth(360)
		num_fps_label.setFixedWidth(360)

		accept_btn = QPushButton('OK')
		accept_btn.clicked.connect(self.getAllOpts)

		# Display text
		# with default values
		self.tft_model_dtxt = QLineEdit()
		self.tft_model_dtxt.setText('../results/tft/latest_net_TFT.pth')
		tft_model_action = self.tft_model_dtxt.addAction(QIcon('../resources/open.png'), self.tft_model_dtxt.TrailingPosition)
		tft_model_action.triggered.connect(lambda: self.openFileNameDialog())

		self.transition_mode_comboBox = QComboBox()
		self.transition_mode_comboBox.addItems(transition_modes)
		self.transition_mode_comboBox.setCurrentIndex(0)
		self.transition_mode_comboBox.currentIndexChanged.connect(self.onCurrentIndexChanged)

		self.num_inbetween_dtxt = QLineEdit()
		# setting background color to the Qlineedit widget
		self.num_inbetween_dtxt.setStyleSheet("QLineEdit"
        									  "{"
        									  "background : white;"
        									  "}")
		self.num_inbetween_dtxt.setText('10')

		self.num_fps_dtxt = QLineEdit()
		self.num_fps_dtxt.setStyleSheet("QLineEdit"
									"{"
									"background : lightgrey;"
									"}")

		self.tft_model_dtxt.setFixedHeight(50)
		self.transition_mode_comboBox.setFixedHeight(50)
		self.num_inbetween_dtxt.setFixedHeight(50)
		self.num_fps_dtxt.setFixedHeight(50)

		self.tft_model_dtxt.setFixedWidth(360)
		self.transition_mode_comboBox.setFixedWidth(360)
		self.num_inbetween_dtxt.setFixedWidth(360)
		self.num_fps_dtxt.setFixedWidth(360)

		self.tft_model_dtxt.setReadOnly(False)
		self.num_inbetween_dtxt.setReadOnly(False)
		self.num_fps_dtxt.setReadOnly(True)

		# Layout
		tft_model_hbox = QHBoxLayout()
		tft_model_hbox.addWidget(tft_model_label)
		tft_model_hbox.addWidget(self.tft_model_dtxt)

		transition_mode_hbox = QHBoxLayout()
		transition_mode_hbox.addWidget(transition_mode_label)
		transition_mode_hbox.addWidget(self.transition_mode_comboBox)
		
		num_inbetween_hbox = QHBoxLayout()
		num_inbetween_hbox.addWidget(num_inbetween_label)
		num_inbetween_hbox.addWidget(self.num_inbetween_dtxt)

		num_fps_hbox = QHBoxLayout()
		num_fps_hbox.addWidget(num_fps_label)
		num_fps_hbox.addWidget(self.num_fps_dtxt)

		layout = QVBoxLayout()
		layout.addLayout(tft_model_hbox)
		layout.addLayout(transition_mode_hbox)
		layout.addLayout(num_inbetween_hbox)
		layout.addLayout(num_fps_hbox)
		layout.addWidget(accept_btn)
		self.setLayout(layout)

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def onCurrentIndexChanged(self):
		if self.transition_mode_comboBox.currentIndex() == 0:
			self.num_inbetween_dtxt.clear()
			self.num_inbetween_dtxt.setReadOnly(False)
			self.num_inbetween_dtxt.setStyleSheet("QLineEdit"
        									      "{"
        									  	  "background : white;"
        									  	  "}")

			self.num_fps_dtxt.clear()
			self.num_fps_dtxt.setReadOnly(True)
			self.num_fps_dtxt.setStyleSheet("QLineEdit"
        									"{"
        									"background : lightgrey;"
        									"}")
		else:
			self.num_inbetween_dtxt.clear()
			self.num_inbetween_dtxt.setReadOnly(True)
			self.num_inbetween_dtxt.setStyleSheet("QLineEdit"
        									      "{"
        									  	  "background : lightgrey;"
        									  	  "}")

			self.num_fps_dtxt.clear()
			self.num_fps_dtxt.setReadOnly(False)
			self.num_fps_dtxt.setStyleSheet("QLineEdit"
											"{"
											"background : white;"
											"}")

	def openFileNameDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		name, _ = QFileDialog.getOpenFileName(self,"Select model", "../results","All Files (*);;PTH Files (*.pth)", options=options)
		self.tft_model_dtxt.setText(name)

	def showMessage(self):
		QMessageBox.about(self, 'Warning', 'Opts cannot be empty!')

	def getAllOpts(self):
		params = []

		tft_model = self.tft_model_dtxt.text()
		transition_mode_ID = self.transition_mode_comboBox.currentIndex()
		if self.transition_mode_comboBox.currentIndex() == 0:
			num_inbetween = self.num_inbetween_dtxt.text()
			num_fps = -1
		else:
			num_inbetween = -1
			num_fps = self.num_fps_dtxt.text()

		params.append(tft_model)
		params.append(transition_mode_ID)
		params.append(num_inbetween)
		params.append(num_fps)

		if len(tft_model) == 0:
			self.showMessage()
		else:
			self.procDone.emit(params)
			self.close()


class SelectWindow(QWidget):
	procDone = QtCore.Signal(list)
	def __init__(self):
		super(SelectWindow, self).__init__()
		self.initUI()

	def initUI(self):
		self.resize(2000, 1200)
		self.center()
		self.setWindowTitle('Select Frames')

		#Past widgets
		past_label = QLabel('Past')
		past_label.setFixedHeight(25)
		past_label.setFixedWidth(150)

		self.past_dtxt = QLineEdit()
		past_dtxt_action = self.past_dtxt.addAction(QIcon('../resources/open.png'), self.past_dtxt.TrailingPosition)
		past_dtxt_action.triggered.connect(lambda: self.openFileNameDialogP())

		past_len_label = QLabel('Len: 10')
		past_len_label.setFixedHeight(25)
		past_len_label.setFixedWidth(150)

		self.GLViewerL = gl.GLViewWidget()
		self.slL = QSlider(Qt.Horizontal)
		self.slL.sliderReleased.connect(self.updateLGL)
		self.slL.valueChanged.connect(self.updateslLLabel)
		self.slLLabel = QLabel()
		self.slLLabel.setFixedHeight(25)
		self.slLLabel.setFixedWidth(150)

		past_prev_btn = QPushButton('Prev')
		past_prev_btn.clicked.connect(self.prevActionL)
		past_next_btn = QPushButton('Next')
		past_next_btn.clicked.connect(self.nextActionL)

		#Target widgets
		target_label = QLabel('Target')
		target_label.setFixedHeight(25)
		target_label.setFixedWidth(150)

		self.target_dtxt = QLineEdit()
		target_dtxt_action = self.target_dtxt.addAction(QIcon('../resources/open.png'), self.target_dtxt.TrailingPosition)
		target_dtxt_action.triggered.connect(lambda: self.openFileNameDialogT())

		target_len_label = QLabel('Len: 1')
		target_len_label.setFixedHeight(25)
		target_len_label.setFixedWidth(150)

		self.GLViewerR = gl.GLViewWidget()
		self.slR = QSlider(Qt.Horizontal)
		self.slR.sliderReleased.connect(self.updateRGL)
		self.slR.valueChanged.connect(self.updateslRLabel)
		self.slRLabel = QLabel()
		self.slRLabel.setFixedHeight(25)
		self.slRLabel.setFixedWidth(150)

		target_prev_btn = QPushButton('Prev')
		target_prev_btn.clicked.connect(self.prevActionR)
		target_next_btn = QPushButton('Next')
		target_next_btn.clicked.connect(self.nextActionR)

		#layout left
		past_txt_hbox = QHBoxLayout()
		past_txt_hbox.addWidget(past_label)
		past_txt_hbox.addWidget(self.past_dtxt)

		past_btn_hbox = QHBoxLayout()
		past_btn_hbox.addWidget(past_prev_btn)
		past_btn_hbox.addWidget(past_next_btn)

		left_vbox = QVBoxLayout()
		left_vbox.addLayout(past_txt_hbox)
		left_vbox.addWidget(past_len_label)
		left_vbox.addWidget(self.GLViewerL)
		left_vbox.addWidget(self.slL)
		left_vbox.addWidget(self.slLLabel)
		left_vbox.addLayout(past_btn_hbox)

		#layout right
		target_txt_hbox = QHBoxLayout()
		target_txt_hbox.addWidget(target_label)
		target_txt_hbox.addWidget(self.target_dtxt)

		target_btn_hbox = QHBoxLayout()
		target_btn_hbox.addWidget(target_prev_btn)
		target_btn_hbox.addWidget(target_next_btn)

		right_vbox = QVBoxLayout()
		right_vbox.addLayout(target_txt_hbox)
		right_vbox.addWidget(target_len_label)
		right_vbox.addWidget(self.GLViewerR)
		right_vbox.addWidget(self.slR)
		right_vbox.addWidget(self.slRLabel)
		right_vbox.addLayout(target_btn_hbox)

		bottom_hbox = QHBoxLayout()
		accept_btn = QPushButton('OK')
		cancel_btn = QPushButton('Cancel')
		accept_btn.clicked.connect(self.acceptAction)
		cancel_btn.clicked.connect(self.close)
		bottom_hbox.addWidget(accept_btn)
		bottom_hbox.addWidget(cancel_btn)

		layout_top = QHBoxLayout()
		layout_top.addLayout(left_vbox)
		layout_top.addLayout(right_vbox)

		layout = QVBoxLayout()
		layout.addLayout(layout_top)
		layout.addLayout(bottom_hbox)
		self.setLayout(layout)

		self.readyL = False
		self.readyR = False

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def acceptAction(self):
		try:
			past_idx = list(range(self.current_sll_idx - 1, self.current_sll_idx - 1 + 11))
			target_idx = self.current_slr_idx - 1
			self.procDone.emit([self.past_name, past_idx, self.target_name, target_idx])
			self.close()
		except:
			msg = QMessageBox()
			msg.setIcon(QMessageBox.Critical)
			msg.setText("Didn't select past or target!")
			msg.setWindowTitle("Error")
			msg.exec_()

	def prevActionL(self):
		if self.readyL:
			current_item_l = self.current_sll_idx - 1
			self.current_sll_idx = (current_item_l - 1) % self.size_l + 1
			self.slL.setValue(self.current_sll_idx)
			self.GLViewerL.clear()
			self.drawItem(True)

	def nextActionL(self):
		if self.readyL:
			current_item_l = self.current_sll_idx - 1
			self.current_sll_idx = (current_item_l + 1) % self.size_l +1
			self.slL.setValue(self.current_sll_idx)
			self.GLViewerL.clear()
			self.drawItem(True)

	def prevActionR(self):
		if self.readyR:
			current_item_r = self.current_slr_idx - 1
			self.current_slr_idx = (self.current_item_r - 1) % self.size_r + 1
			self.slR.setValue(self.current_slr_idx)
			self.GLViewerR.clear()
			self.drawItem(False)

	def nextActionR(self):
		if self.readyR:
			current_item_r = self.current_slr_idx - 1
			self.current_slr_idx = (self.current_item_r + 1) % self.size_r + 1
			self.slR.setValue(self.current_slr_idx)
			self.GLViewerR.clear()
			self.drawItem(False)

	def openFileNameDialogP(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.past_name, _ = QFileDialog.getOpenFileName(self,"Select past poses", "../dataset/lafan/test_set","All Files (*);;PKL Files (*.pkl)", options=options)
		self.past_dtxt.setText(self.past_name)
		self.getItems(self.past_name, True)
		self.readyL = True

	def openFileNameDialogT(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.target_name, _ = QFileDialog.getOpenFileName(self,"Select target pose", "../dataset/lafan/test_set","All Files (*);;PKL Files (*.pkl)", options=options)
		self.target_dtxt.setText(self.target_name)
		self.getItems(self.target_name, False)
		self.readyR = True


	def getItems(self, name, left):
		with open(name, 'rb') as f:
			data = pickle.load(f, encoding='latin1')
		x_local = data['x_local']
		q_local = data['q_local']
		size = x_local.shape[0]
		if left:
			self.size_l = size
			self.slL.setRange(1, self.size_l)
			self.current_sll_idx = 1
			_,self.xl = quat_fk(q_local, x_local, parents)
			self.drawItem(left)
		else:
			self.size_r = size
			self.slR.setRange(1, self.size_r)
			self.current_slr_idx = 1
			_,self.xr = quat_fk(q_local, x_local, parents)
			self.drawItem(left)

	def drawItem(self, left):
		# init GLViewer
		r = R.from_euler('xyz', [90, 0, 180], degrees=True)
		rr = r.as_matrix()
		data = self.xl[self.current_sll_idx-1].copy() if left else self.xr[self.current_slr_idx-1].copy()
		data = norm_x(data)
		data = np.matmul(data, rr)
		if left:
			self.GLViewerL.opts['distance'] = 2.5
			self.GLViewerL.opts['elevation'] = 20
			self.GLViewerL.opts['azimuth'] = 50
			self.GLViewerL.setBackgroundColor('w')
		else:
			self.GLViewerR.opts['distance'] = 2.5
			self.GLViewerR.opts['elevation'] = 20
			self.GLViewerR.opts['azimuth'] = 50
			self.GLViewerR.setBackgroundColor('w')
		for i in range(1, 22):
			xx = (data[i][0], data[i][1], data[i][2])
			yy = (data[parents[i]][0], data[parents[i]][1], data[parents[i]][2])
			pts = np.array([xx, yy])

			center = (data[i] + data[parents[i]]) / 2.
			length = np.linalg.norm(data[i] - data[parents[i]]) / 2.
			radius = [0.02, 0.02]
			md = gl.MeshData.cylinder(rows=40, cols=40, radius=radius, length=2*length)

			m1 = gl.GLMeshItem(meshdata=md,
							   smooth=True,
							   color=(0.81, 0.13, 0.56, 1),
							   shader="shaded",
							   glOptions="opaque")
			
			v = data[i] - data[parents[i]]
			theta = np.arctan2(v[1], v[0])
			phi = np.arctan2(np.linalg.norm(v[:2]), v[2])

			tr = pg.Transform3D()
			tr.translate(*data[parents[i]])
			tr.rotate(theta * 180 / np.pi, 0, 0, 1)
			tr.rotate(phi * 180 / np.pi, 0, 1, 0)
			tr.scale(1, 1, 1)
			tr.translate(0, 0, 0)
			m1.setTransform(tr)

			if left:
				self.GLViewerL.addItem(m1)
			else:
				self.GLViewerR.addItem(m1)

		im = np.asarray(Image.open('../resources/checkerboard.png'))
		tex = pg.makeRGBA(im)[0]
		image = gl.GLImageItem(tex, glOptions='opaque')
		image.translate(-im.shape[0]/2., -im.shape[1]/2., -1)
		if left:
			self.GLViewerL.addItem(image)
		else:
			self.GLViewerR.addItem(image)


	def updateLGL(self):
		self.current_sll_idx = self.slL.value()
		self.GLViewerL.clear()
		self.drawItem(True)

	def updateRGL(self):
		self.current_slr_idx = self.slR.value()
		self.GLViewerR.clear()
		self.drawItem(False)

	def updateslLLabel(self):
		self.slLLabel.setText(str(self.slL.value()))

	def updateslRLabel(self):
		self.slRLabel.setText(str(self.slR.value()))


class dmtWrapper(QWidget):
	"""Wapper for DMT functions
	"""
	procUpdate = QtCore.Signal()
	def __init__(self):
		super(dmtWrapper, self).__init__()
		self.tft_model_path = '../results/tft/latest_net_TFT.pth'
		self.transition_mode = 0
		self.num_inbetween = 10
		self.fps = 30

		self.dmp_q = dmpQ()
		self.dmp_p = dmpP()


	def init_opt(self):
		opt = TestOptions().parse()
		opt.name = 'tft'
		opt.model = 'tft'
		opt.checkpoints_dir = '../results'
		opt.num_joints = 22
		opt.hidden_size = 128
		opt.past_len = 10
		model_name = os.path.split(self.tft_model_path)[1]
		endPos = model_name.find('_')
		opt.epoch = model_name[:endPos]

		return opt


	def run_dmt(self):
		"""Run DMT given trained TFT model
		Parameters:
			past: ndarray of past frames
			target: target frame
		"""
		#init tft
		opt = self.init_opt()
		tft = create_model(opt)
		tft.setup(opt)
		tft.eval()

		# get input data for DMT
		force_past = self.past[self.past_idx_plus]['force']
		quat_past = self.past[self.past_idx_plus]['q_local']
		quat_target = self.target[self.target_idx]['q_local']
		x_past = self.past[self.past_idx_plus]['x_local']
		x_target = self.target[self.target_idx]['x_local']
		lin_v_past = self.past[self.past_idx_plus]['lin_v']
		lin_v_target = self.target[self.target_idx]['lin_v']
		f_inp = force_past[:-1]
		q_inp = quat_past

		_, x_past_glbl = quat_fk(quat_past, x_past, parents)
		_, x_target_glbl = quat_fk(quat_target, x_target, parents)

		#compute number of frames in between
		# past has length of 10 + 1 = 11, n_force=10, n_quat=11
		v_s = lin_v_past[-1][0]
		v_e = lin_v_target[0]
		v_avg = (v_s + v_e) / 2.
		x_s = x_past_glbl[-1][0]
		x_e = x_target_glbl[0]
		x_inbetween = x_e - x_s
		t_duration = x_inbetween / v_avg
		num_inbetween = t_duration * self.fps

		q = []
		q.append(q_inp[-1])
		p = []
		p.append(x_s)

		# iterate all the in-between motions
		for i in range(num_inbetween):
			inp = {'quat': q_inp, 'force': f_inp}
			tft.set_inference_input(inp)
			f = tft.run_step()

			torque = f.copy()
			torque[0] = [0.,0.,0.]
			force = f.copy()
			force = force[0]

			# 1. quaternion from torque
			q_next = self.dmp_q.compute_next_state(torque, quat_target)

			# 2. global position from force
			p_next = self.dmp_p.compute_next_state(force, x_e)

			q.append(q_next)
			p.append(p_next)

			# update f input and q input
			f_inp = np.concatenate(f_inp[1:], f[np.newaxis, ...], axis=0)
			q_inp = np.concatenate(q_inp[1:], q_next[np.newaxis, ...], axis=0)

		# 1. rotation
		q = np.stack(q, axis=0)
		rot = quat2euler(q)

		# 2. global offset
		off = np.stack(p, axis=0)
		off = off - x_past_glbl[0][0]
		off[:, 1] = 0

		rot_target = self.rots[-1]
		off_target = self.offs[-1]
		self.rots = np.concatenate((self.rots[:-1], rot), axis=0)
		self.offs = np.concatenate((self.offs[:-1], off), axis=0)

		self.rots = np.concatenate((self.rots, rot_target), axis=0)
		self.offs = np.concatenate((self.offs, off_target), axis=0)
		self.updateViewer()


	def updateViewer(self):
		"""
		Update main viewer
		"""
		self.procUpdate.emit()

	@QtCore.Slot(list)
	def getPastTarget(self, val):
		"""
        Get past and target frames from select window
        """
		past_path = val[0]
		self.past_idx = val[1]
		self.past_idx_plus = self.past_idx[:-1]
		target_path = val[2]
		self.target_idx = val[3]

		with open(past_path, 'rb') as f:
			self.past = pickle.load(f, encoding='latin1')
		with open(target_path, 'rb') as f:
			self.target = pickle.load(f, encoding='latin1')

		q_local_past = self.past['q_local'][self.past_idx_plus]
		x_local_past = self.past['x_local'][self.past_idx_plus]
		q_local_target = np.expand_dims(self.target['q_local'][self.target_idx], axis=0)
		x_local_target = np.expand_dims(self.target['x_local'][self.target_idx], axis=0)

		_, off_past = quat_fk(q_local_past, x_local_past, parents)
		off_past = off_past[:, 0, :]
		off_past = (off_past - off_past[0]) * 1
		_, off_target = quat_fk(q_local_target, x_local_target, parents)
		off_target = off_target[:, 0, :]
		off_target = (off_target - off_past[0]) * 1

		# set offset y to 0
		off_past[:, 1] = 0
		off_target[:, 1] = 0

		rot_past = quat2euler(q_local_past)
		rot_target = quat2euler(q_local_target)

		self.rots = np.concatenate((rot_past, rot_target), axis=0)
		self.offs = np.concatenate((off_past, off_target), axis=0)
		self.updateViewer()

	@QtCore.Slot(list)
	def getInBetween(self, val):
		"""
		Get past and target frames from select window
		"""
		past_path = val[0]
		self.past_idx = val[1]
		target_path = val[2]
		self.target_idx = val[3]

		with open(past_path, 'rb') as f:
			self.past = pickle.load(f, encoding='latin1')
		with open(target_path, 'rb') as f:
			self.target = pickle.load(f, encoding='latin1')


	@QtCore.Slot(list)
	def getSetting(self, params):
		"""
		Get settings from dmtSetting window
		"""
		self.tft_model_path = params[0]
		self.transition_mode = params[1]
		self.num_inbetween = params[2]
		self.num_fps = params[3]



class GLWidget(gl.GLViewWidget):
	def __init__(self, selectTarget=False):
		gl.GLViewWidget.__init__(self)
		self.selectTarget = selectTarget
		self.targetOldPos = None
		self.targetNewPos = None

	def mousePressEvent(self, ev):
		if not self.selectTarget:
			lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
			self.mousePos = lpos
		else:
			verts = self.items[-1].opts['meshdata'].vertexes()
			self.targetOldPos = np.mean(verts, axis=0)

	def mouseMoveEvent(self, ev):
		if not self.selectTarget:
			lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
			diff = lpos - self.mousePos
			self.mousePos = lpos

			if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
				if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
					self.pan(diff.x(), diff.y(), 0, relative='view')
				else:
					self.orbit(-diff.x(), diff.y())
			elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
				if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
					self.pan(diff.x(), 0, diff.y(), relative='view-upright')
				else:
					self.pan(diff.x(), diff.y(), 0, relative='view-upright')
		else:
			pass



	def mouseReleaseEvent(self, ev):
		if not self.selectTarget:
			pass
		else:
			x0, y0, w, h = self.getViewport()
			dist = self.opts['distance']
			fov = self.opts['fov']
			nearClip = dist * 0.001
			farClip = dist * 1000.
			lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
			self.mousePos = lpos

			r = nearClip * math.tan(0.5 * math.radians(fov))
			t = r * h / w

			view_matrix = self.viewMatrix()
			matrix_inv = view_matrix.inverted()[0]

			val = np.zeros(3)
			val[0] = (2.0 * ((self.mousePos.x() - 0) / (self.deviceWidth() - 0))) - 1.0
			val[1] = 1.0 - (2.0 * ((self.mousePos.y() - 0) / (self.deviceHeight() - 0)))
			val[0] *= r
			val[1] *= t
			val[2] = nearClip

			pp = np.array([val[0], val[1], val[2]])
			z = abs(self.cameraPosition().z()) - abs(self.targetOldPos[2])
			x = pp[0] * z / pp[2]
			y = pp[1] * z / pp[2]

			pos = np.array([[x], [y], [z], [1.]])
			matrix_inv = np.matrix(matrix_inv.copyDataTo())
			matrix_inv = np.reshape(matrix_inv, (4,4))
			pos = np.matmul(matrix_inv, pos)
			pos = np.array([pos.item((0,0)), pos.item((1,0)), pos.item((2,0)), pos.item((3,0))])
			self.targetNewPos = pos[:3]
			self.targetNewPos[2] = self.targetOldPos[2]

			diff = self.targetNewPos - self.targetOldPos
			verts = self.items[-1].opts['meshdata'].vertexes()
			verts = verts + diff
			meshdata = gl.MeshData(vertexes=verts, faces=faces)
			self.items[-1].setMeshData(meshdata=meshdata)



		# Example item selection code:
	# region = (ev.pos().x()-5, ev.pos().y()-5, 10, 10)
	# print(self.itemsAt(region))

	## debugging code: draw the picking region
	# glViewport(*self.getViewport())
	# glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
	# region = (region[0], self.height()-(region[1]+region[3]), region[2], region[3])
	# self.paintGL(region=region)
	# self.swapBuffers()

class Viewer(QMainWindow):
	def __init__(self):
		super(Viewer, self).__init__()
		self.dw = dmtWrapper()
		self.selectWindow = SelectWindow()
		self.dmtSettingWindow = dmtSettingWindow()
		self.selectTarget = False
		self.currentViewID = 0
		self.currentViewParams = []
		self.initUI()

	def initUI(self):
		pg.setConfigOption('background', 'w')
		pg.setConfigOption('foreground', 'k')
		exitAct = QAction(QIcon('exit.png'), '&Exit', self)
		exitAct.setShortcut('Ctrl+Q')
		exitAct.setStatusTip('Exit application')
		exitAct.triggered.connect(qApp.quit)

		dmtSettingAct = QAction('DMT settings', self)
		dmtSettingAct.setShortcut('Ctrl+S')
		dmtSettingAct.setStatusTip('DMT settings')
		dmtSettingAct.triggered.connect(self.showSettingWindow)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(dmtSettingAct)
		fileMenu.addAction(exitAct)

		self.resize(2560, 1200)
		self.center()
		self.setWindowTitle('Motion Transition')

		# top HBox
		top_hbox = QHBoxLayout()
		selectButton = QPushButton('Select Past/Target')
		selectButton.clicked.connect(self.showSelectWindow)
		setSButton = QPushButton('45 view/top view/side view')
		setSButton.clicked.connect(self.toggleCameraView)
		setTButton = QPushButton('Toggle target edit mode')
		setTButton.clicked.connect(self.toggleTargetEdit)
		top_hbox.addWidget(selectButton)
		top_hbox.addWidget(setSButton)
		top_hbox.addWidget(setTButton)

		# Middle HBox
		middle_hbox = QHBoxLayout()
		self.GLViewer = GLWidget()
		middle_hbox.addWidget(self.GLViewer)

		# Bottom HBox
		bottom_hbox = QHBoxLayout()

		# Middle window
		leftButton = QPushButton('Run')
		leftButton.clicked.connect(self.runAction)
		rightButton = QPushButton('Reset')
		rightButton.clicked.connect(self.resetAction)
		middle_bottom_hbox = QHBoxLayout()
		middle_bottom_hbox.addWidget(leftButton)
		middle_bottom_hbox.addWidget(rightButton)

		bottom_hbox.addLayout(middle_bottom_hbox)

		# Set layout
		layout = QVBoxLayout()
		layout.addLayout(top_hbox)
		layout.addLayout(middle_hbox)
		layout.addLayout(bottom_hbox)

		widget = QWidget()
		widget.setLayout(layout)
		self.setCentralWidget(widget)

		self.current_past_item = 0
		self.current_target_item = 0
		self.dw.procUpdate.connect(self.drawItems)

		self.show()

	def closeEvent(self, event):
		for window in QApplication.topLevelWidgets():
			window.close()

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def openTFTModelDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.file, _ = QFileDialog.getOpenFileName(self,"Select TFT model", "../results","All Files (*);;PKL Files (*.pkl)", options=options)
		self.getPostItems(self.file)

	def showSettingWindow(self):
		self.dmtSettingWindow.show()
		self.dmtSettingWindow.procDone.connect(self.dw.getSetting)

	def showSelectWindow(self):
		self.selectWindow.show()
		self.selectWindow.procDone.connect(self.dw.getPastTarget)


	def getSettings(self, opts):
		self.dw.tft_model = opts[0]
		self.dw.transition_mode = opts[1]
		self.dw.sample_rate = opts[2]
		if opts[1] == 1:
			self.dw.num_inbetween = opts[3]
		self.dw.past_len = opts[4]

	def getItems(self, pitems, titem):
		self.pastQuat = []
		self.pastForce = []
		# read past frames
		for i in pitems:
			with open(i, 'rb') as f:
				data = pickle.load(f, encoding='latin1')
			quat = data['q_local']
			force = data['force']
			quat = quat.reshape(quat.shape[0], -1)
			force = force.reshape(force.shape[0], -1)
			self.pastQuat.append(quat)
			self.pastForce.append(force)
		self.pastQuat = np.asarray(self.pastQuat)
		self.pastForce = np.asarray(self.pastForce)
		#resize to [len, batch, feat]
		self.pastQuat = self.pastQuat[:, np.newaxis, :]
		self.pastForce = self.pastForce[:, np.newaxis, :]

		# read target
		with open(titem, 'rb') as f:
			data = pickle.load(f, encoding='latin1')
		self.targetQuat = data['q_local']
		self.targetAngV = data['ang_v']
		self.targetQuat = self.targetQuat.reshape(self.targetQuat.shape[0], -1)
		self.targetAngV = self.targetAngV.reshape(self.targetAngV.shape[0], -1)


	def runAction(self):
		self.current_target_item = (self.current_target_item - 1) % 1
		self.GLViewer.clear()
		self.drawItem()

	def resetAction(self):
		self.current_target_item = (self.current_target_item - 1) % 1
		self.GLViewer.clear()
		self.drawItem()

	@QtCore.Slot()
	def drawItems(self):
		# init GLViewer
		self.GLViewer.clear()
		self.GLViewer.opts['distance'] = 20
		im = np.asarray(Image.open('../resources/checkerboard.png'))
		tex = pg.makeRGBA(im)[0]
		image = gl.GLImageItem(tex, glOptions='opaque')
		image.translate(-im.shape[0] / 2., -im.shape[1] / 2., 0)
		self.GLViewer.addItem(image)

		last = self.dw.rots.shape[0] - 1
		for i in range(self.dw.rots.shape[0]):
			self.drawItem(i, 10, last)


	def drawItem(self, idx, past, last):
		# init GLViewer
		rot = self.dw.rots[idx]
		off = self.dw.offs[idx]
		data = getMayaUpdateMesh(rot, off)

		if idx == 0:
			verts = data.vertexes()
			mass = np.mean(verts, axis=0)
			self.GLViewer.setCameraPosition(pos=QtGui.QVector3D(mass[0], mass[1], mass[2]), elevation=30)
			params = self.GLViewer.cameraParams()
			params['rotation'] = None
			self.currentViewParams.append(params)

		gl.shaders.Shaders.append(gl.shaders.ShaderProgram('pastShader', [
			gl.shaders.VertexShader("""
		            varying vec3 normal;
		            void main() {
		                // compute here for use in fragment shader
		                normal = normalize(gl_NormalMatrix * gl_Normal);
		                gl_FrontColor = gl_Color;
		                gl_BackColor = gl_Color;
		                gl_Position = ftransform();
		            }
		        """),
			gl.shaders.FragmentShader("""
		            varying vec3 normal;
		            void main() {
		                vec4 color = gl_Color;
		                color.x = 0;
		                color.y = (normal.y + 1.0) * 0.5;
		                color.z = (normal.z + 1.0) * 0.5;
		                gl_FragColor = color;
		            }
		        """)
		]))

		gl.shaders.Shaders.append(gl.shaders.ShaderProgram('targetShader', [
			gl.shaders.VertexShader("""
				            varying vec3 normal;
				            void main() {
				                // compute here for use in fragment shader
				                normal = normalize(gl_NormalMatrix * gl_Normal);
				                gl_FrontColor = gl_Color;
				                gl_BackColor = gl_Color;
				                gl_Position = ftransform();
				            }
				        """),
			gl.shaders.FragmentShader("""
				            varying vec3 normal;
				            void main() {
				                vec4 color = gl_Color;
				                color.x = (normal.x + 1.0) * 0.5;
				                color.y = (normal.y + 1.0) * 0.5;
				                color.z = 0;
				                gl_FragColor = color;
				            }
				        """)
		]))

		if idx < past:
			mesh = gl.GLMeshItem(meshdata=data,
						     	 smooth=True,
						     	 shader="pastShader",
						     	 glOptions="opaque")
		if idx == last:
			mesh = gl.GLMeshItem(meshdata=data,
								 smooth=True,
								 shader="targetShader",
								 glOptions="opaque")

		self.GLViewer.addItem(mesh)

	def updateTarget(self, **kwds):
		md = kwds.get('meshdata', None)
		if md is not None:
			self.GLViewer.items[-1].setMeshData(meshdata=md)
		hasEdges = kwds.get('drawEdges', None)
		if hasEdges is not None:
			self.GLViewer.items[-1].opts['drawEdges'] = hasEdges
			self.GLViewer.items[-1].opts.update(kwds)
			self.GLViewer.items[-1].meshDataChanged()
			self.GLViewer.items[-1].update()


	def toggleTargetEdit(self):
		self.selectTarget = not self.selectTarget
		self.GLViewer.selectTarget = not self.GLViewer.selectTarget

		if self.selectTarget:
			verts = self.GLViewer.items[-1].opts['meshdata'].vertexes()
			mass = np.mean(verts, axis=0)
			self.GLViewer.setCameraPosition(pos=QtGui.QVector3D(mass[0], mass[1], mass[2]), elevation=90)
			params = self.GLViewer.cameraParams()
			params['rotation'] = None
			self.currentViewParams.append(params)
			self.updateTarget(drawEdges=True)
		else:
			self.currentViewParams.pop()
			self.GLViewer.setCameraParams(**self.currentViewParams[-1])
			self.updateTarget(drawEdges=False)


	def toggleCameraView(self):
		"""
		toggle camera view. 45 view: 0; top view: 1; side view: 2
		"""
		self.currentViewID = (self.currentViewID + 1) % 3

		verts = self.GLViewer.items[1].opts['meshdata'].vertexes()
		mass = np.mean(verts, axis=0)

		if self.currentViewID == 0:
			self.GLViewer.setCameraPosition(pos=QtGui.QVector3D(mass[0], mass[1], mass[2]), elevation=30)
		if self.currentViewID == 1:
			self.GLViewer.setCameraPosition(pos=QtGui.QVector3D(mass[0], mass[1], mass[2]), elevation=90)
		if self.currentViewID == 2:
			self.GLViewer.setCameraPosition(pos=QtGui.QVector3D(mass[0], mass[1], mass[2]), elevation=0)
		params = self.GLViewer.cameraParams()
		params['rotation'] = None
		self.currentViewParams[-1] = params

	def generate(self):
		self.pastLatentItems, self.targetLatentItems = self.sw.run_fvae(self.pastOriginalItems, self.targetOriginalItems)
		self.transitions = self.sw.run_rtn(self.pastLatentItems, self.targetLatentItems)


def main():
	app = QApplication(sys.argv)
	viewer = Viewer()
	viewer.show()
	mayaInit()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()