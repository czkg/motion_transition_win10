import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from torch import Tensor
from typing import Dict, Tuple, Optional, List

from torch.nn import LayerNorm


###############################################################################
# Helper functions
###############################################################################

def init_weights(net, init_type='normal', init_gain=1.):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'zeros':
                init.zeros_(m.weight.data)
            elif init_type == 'ones':
                init.ones_(m.weight.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=1., gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def load_net(net, checkpoint, name, epoch, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    load_weights(net, checkpoint, name, epoch, gpu_ids)
    return net


def load_weights(net, checkpoint, name, epoch, gpu_ids):
    net = net.module
    save_dir = os.path.join(checkpoint, name)
    load_filename = '%s_net_%s.pth' % (epoch, name.upper())
    load_path = os.path.join(save_dir, load_filename)
    print('loading the model from %s' % load_path)
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    state_dict = torch.load(load_path, map_location=str(device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    net.load_state_dict(state_dict)


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='none'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


##############################################################################
# Classes
##############################################################################

class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden_size, eps=eps)
    
    def forward(self, x):
        return self.ln(x)


class GLU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size*2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class SGRN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 context_hidden_size=None,
                 dropout=0,
                 scale=0.6):
        super().__init__()

        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None
        self.scale = scale

    def forward(self, a: Tensor, c: Optional[Tensor]=None):
        x = self.lin_a(a)
        if c is not None:
            x = self.scale*x + (1.-self.scale)*self.lin_c(c)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if not self.out_proj else self.out_proj(a)
        x = x + y
        x = self.layer_norm(x)
        return x


class TFTEmbedding(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # There are 2 inputs:
        # 1. Temporal forces
        # 2. Temporal quaternions

        self.f_embedding = nn.Linear(opt.force_size, opt.hidden_size)
        self.q_embedding = nn.Linear(opt.quat_size, opt.hidden_size)


    def forward(self, x: Dict[str, Tensor]):
        # get force and quaternion
        quat = x['quat']
        force = x['force']

        q_inp = self.q_embedding(quat)
        f_inp = self.f_embedding(force)

        return q_inp, f_inp


class VariableSelectionNetwork(nn.Module):
    def __init__(self, opt, num_inputs):
        super().__init__()
        self.joint_grn = SGRN(opt.hidden_size*num_inputs, opt.hidden_size, output_size=num_inputs, context_hidden_size=opt.hidden_size)
        self.var_grns = nn.ModuleList([SGRN(opt.hidden_size, opt.hidden_size, dropout=opt.dropout) for _ in range(num_inputs)])

    def forward(self, x: Tensor, context: Optional[Tensor]=None):
        Xi = x.reshape(*x.shape[:-2], -1)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[...,i,:]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)

        #the line below performs batched matrix vector multiplication
        #for temporal features it's bthf,btf->bth
        #for static features it's bhf,bf->bh
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)

        return variable_ctx, sparse_weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.n_head = opt.n_head
        assert opt.hidden_size % opt.n_head == 0
        self.d_head = opt.hidden_size // opt.n_head
        self.qkv_linears = nn.Linear(opt.hidden_size, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, opt.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(opt.attn_dropout)
        self.out_dropout = nn.Dropout(opt.dropout)
        self.scale = self.d_head**-0.5

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        t, bs, h_size = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(t, bs, self.n_head, self.d_head)
        k = k.view(t, bs, self.n_head, self.d_head)
        v = v.view(t, bs, self.d_head)

        # attn_score = torch.einsum('bind,bjnd->bnij', q, k)
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)

        # attn_vec = torch.einsum('bnij,bjd->bnid', attn_prob, v)
        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)

        return out, attn_vec


class TemporalFusionTransformer(nn.Module):
    """ 
    Implementation of https://arxiv.org/abs/1912.09363 
    """
    def __init__(self, opt):
        super().__init__()

        self.past_len = opt.past_len #this determines from how distant past we want to use data from

        self.embedding = TFTEmbedding(opt)
        self.f_encoder = nn.LSTM(opt.hidden_size, opt.hidden_size, batch_first=False)
        self.q_encoder = nn.LSTM(opt.hidden_size, opt.hidden_size, batch_first=False)

        self.input_gate = GLU(opt.hidden_size, opt.hidden_size)
        self.input_gate_ln = LayerNorm(opt.hidden_size, eps=1e-3)

        self.positionwise_grn = SGRN(opt.hidden_size,
                                     opt.hidden_size, 
                                     dropout=opt.dropout)
        self.attention = InterpretableMultiHeadAttention(opt)
        self.attention_gate = GLU(opt.hidden_size, opt.hidden_size)
        self.attention_ln = LayerNorm(opt.hidden_size, eps=1e-3)

        self.enrichment_grn = SGRN(opt.hidden_size,
                                   opt.hidden_size,
                                   context_hidden_size=opt.hidden_size,
                                   dropout=opt.dropout)
        # self.enrichment_grn = SGRN(opt.hidden_size,
        #                            opt.hidden_size,
        #                            dropout=opt.dropout)

        self.decoder_gate = GLU(opt.hidden_size, opt.hidden_size)
        self.decoder_ln = LayerNorm(opt.hidden_size, eps=1e-3)

        self.proj = nn.Linear(opt.hidden_size, opt.force_size)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        # Temporal input
        q_emb, f_emb = self.embedding(x)

        # Encoders for f and q
        f_features,_ = self.f_encoder(f_emb)
        q_features,_ = self.q_encoder(q_emb)
        q_features = q_features[-1,...]
        torch.cuda.synchronize() # this call gives perf boost for unknown reasons

        # skip connection for f
        f_features = self.input_gate(f_features)
        f_features = f_emb + f_features
        f_features = self.input_gate_ln(f_features)

        # GRN
        f_grn_features = self.positionwise_grn(f_features)

        # Temporal self attention
        x, _ = self.attention(f_grn_features)

        # Don't compute past quantiles
        x = x[-1, ...]
        f_features = f_features[-1, ...]
        f_grn_features = f_grn_features[-1, ...]

        x = self.attention_gate(x)
        x = x + f_grn_features
        x = self.attention_ln(x)

        # Enrichedment GRN
        x = self.enrichment_grn(x, c=q_features)
        #x = self.enrichment_grn(x)

        # Final skip connection
        x = self.decoder_gate(x)
        x = x + f_features
        x = self.decoder_ln(x)

        out = self.proj(x)

        return out


class TFTRecLoss(nn.Module):
    def __init__(self):
        super(TFTRecLoss, self).__init__()

    def __call__(self, predictions, targets):
        """
        Parameters:
            predictions: predicted forces of length seq_length
            target: ground truth forces of length seq_length
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(predictions, targets, reduction='sum')

        return recon_loss


class TFTPlaLoss(nn.Module):
    def __init__(self):
        super(TFTPlaLoss, self).__init__()

    def __call__(self, predictions, targets, lin_a, mass=1.):
        """
        Parameters:
            predictions: predicted forces of length seq_length
            target: ground truth forces of length seq_length
            lin_a: linear accelaration of length seq_length
            mass: default mass as 1.
        """
        # PLA (principle of least action) loss
        pla_loss = F.mse_loss(predictions, mass*lin_a)

        return pla_loss