import torch
from .base_model import BaseModel
from . import networks


class TFTModel(BaseModel):
	""" This class implements the TFT model.
	"""

	@staticmethod
	def modify_commandline_options(parser, is_train = True):
		"""Add new options
		"""
		return parser


	def __init__(self, opt):
		""" Initialize the vae class.

		Parameters:
			opt (Option class)-- stores all the experiment flags, needs to be a subclass of BaseOptions
		"""
		BaseModel.__init__(self, opt)
		self.loss_names = ['TFTRec']
		self.model_names = ['TFT']
		self.netTFT = networks.TemporalFusionTransformer(opt)
		self.netTFT = networks.init_net(self.netTFT, init_type = opt.init_type, init_gain = opt.init_gain, gpu_ids = opt.gpu_ids)
		self.past_len = opt.past_len
		if self.isTrain:
			#define loss functions
			self.criterionTFTRec = networks.TFTRecLoss().to(self.device)
			#self.criterionTFTPla = networks.TFTPlaLoss().to(self.device)
			#initialize optimizers
			self.optimizerTFT = torch.optim.Adam(self.netTFT.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999), eps = 1e-6)
			self.optimizers.append(self.optimizerTFT)



	def set_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		quat = input['q_local'].to(self.device).float()
		force = input['force'].to(self.device).float()
		lin_a = input['lin_a'].to(self.device).float()
		#convert [batch, sequence, feature] to [sequence, batch, feature]
		quat = torch.permute(quat, (1,0,2))
		force = torch.permute(force, (1,0,2))
		lin_a = torch.permute(lin_a, (1,0,2))

		quat_inp = quat[:self.past_len+1, ...]
		force_inp = force[:self.past_len, ...]
		self.input = {'quat': quat_inp, 'force': force_inp}
		self.target = force[-1, ...]
		self.lin_a = lin_a[-1,...]
		self.file_name = input['info']


	def set_inference_input(self, input):
		""" Unpack input data from the dataloader and perform necessary pre-processing steps.
		Parameters:
			input (dict): include the data itself and its metadata information.
		"""
		quat = torch.tensor(input['quat'])
		force = torch.tensor(input['force'])
		quat = quat.to(self.device).float()
		force = force.to(self.device).float()

		self.input = {'quat': quat, 'force': force}


	def get_model(self):
		return self.netTFT

	def run_step(self):
		"""Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
		with torch.no_grad():
			self.forward()
		return self.output.cpu().detach().numpy()


	def forward(self):
		""" Run forward pass, called by both functions <optimize_parameters> and <test>
		"""
		self.output = self.netTFT(self.input)


	def update(self):
		self.set_requires_grad(self.netTFT, True)  # enable backprop
		self.optimizerTFT.zero_grad()              # set gradients to zero

		self.loss_TFTRec = self.criterionTFTRec(self.output, self.target)
		#self.loss_TFTPla = self.criterionTFTPla(self.output, self.target, self.lin_a)
		self.loss_TFT = self.loss_TFTRec
		self.loss_TFT.backward()

		self.optimizerTFT.step()


	def get_current_out_in(self):
		return self.output[0],self.input[0]


	def optimize_parameters(self):
		self.forward()
		self.update()