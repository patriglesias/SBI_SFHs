import numpy as np
import torch
from torch import nn
#from torchinterp1d import Interp1d

class MLP(nn.Sequential):
    """Multi-Layer Perceptron

    A simple implementation with a configurable number of hidden layers and
    activation functions.

    Parameters
    ----------
    n_in: int
        Input dimension
    n_out: int
        Output dimension
    n_hidden: list of int
        Dimensions for every hidden layer
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(16, 16, 16),
                 act=None,
                 dropout=0):

        if act is None:
            act = [ nn.LeakyReLU(), ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_)-1):
                layer.append(nn.Linear(n_[i], n_[i+1]))
                layer.append(act[i])
                layer.append(nn.Dropout(p=dropout))

        super(MLP, self).__init__(*layer)

class SpectrumEncoder(nn.Module):
    """Spectrum encoder

    Modified version of the encoder by Serrà et al. (2018), which combines a 3 layer CNN
    with a dot-product attention module. This encoder adds a MLP to further compress the
    attended values into a low-dimensional latent space.

    Paper: Serrà et al., https://arxiv.org/abs/1805.03908

    Parameters
    ----------
    instrument: :class:`spender.Instrument`
        Instrument that observed the data
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the :class:`MLP`
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    n_aux: int
        Dimensions of auxiliary inputs for the :class:`MLP`
    dropout: float
        Dropout probability
    """
    def __init__(self,
                 instrument,
                 n_latent,
                 n_hidden=(128, 64, 32),
                 act=None,
                 n_aux=0,
                 dropout=0):

        super(SpectrumEncoder, self).__init__()
        self.instrument = instrument
        self.n_latent = n_latent
        self.n_aux = n_aux

        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1] // 2

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s, padding=s//2) for s in sizes[:2])
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features + aux to latents
        if act is None:
            act = [ nn.PReLU(n) for n in n_hidden ]
            # last activation identity to have latents centered around 0
            act.append(nn.Identity())
        self.mlp = MLP(self.n_feature + n_aux, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout)


    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(len(filters)):
            f_in = 1 if i == 0 else filters[i-1]
            f = filters[i]
            s = sizes[i]
            p = s // 2
            conv = nn.Conv1d(in_channels=f_in,
                             out_channels=f,
                             kernel_size=s,
                             padding=p,
                            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def _downsample(self, x):
        # compression
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        C = x.shape[1] // 2
        # split half channels into attention value and key
        h, a = torch.split(x, [C, C], dim=1)

        return h, a

    def forward(self, y, aux=None):
        """Forward method

        Parameters
        ----------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra
        aux: `torch.tensor`, shape (N, n_aux)
            (optional) Batch of auxiliary inputs to MLP

        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        # run through CNNs
        h, a = self._downsample(y)
        # softmax attention
        a = self.softmax(a)

        # attach hook to extract backward gradient of a scalar prediction
        # for Grad-FAM (Feature Activation Map)
        if ~self.training and a.requires_grad == True:
            a.register_hook(self._attention_hook)

        # apply attention
        x = torch.sum(h * a, dim=2)

        #final MLP
        x = self.mlp(x)
        return x

    @property
    def n_parameters(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _attention_hook(self, grad):
        self._attention_grad = grad

    @property
    def attention_grad(self):
        """Gradient of the attention weights

        Factor to compute the importance of attention for Grad-FAM method.

        Requires a previous `loss.backward` call for any scalar loss function based on
        outputs of this classes `forward` method. This functionality is switched off
        during training.
        """
        if hasattr(self, '_attention_grad'):
            return self._attention_grad
        else:
            return None


#by Patricia, to generate latents optimized for getting percentiles

class Base_encoder_percentiles(nn.Module):
    """Base class for spectrum encoder optimized for obtaining percentiles

    This class is agnostic about the encoder and MLP architectures. It simply calls
    them in order and computes the loss for the recontruction fidelity.

    The only requirements for the modules is that they have the same latent
    dimensionality, and for the 'loss' method the length of the observed spectrum
    vectors need to agree.

    Parameter
    ---------
    encoder: `nn.Module`
        Encoder
    MLP: `nn.Module`
        MLP
    """
    def __init__(self,
                 encoder,
                 mlp,
                ):

        super(Base_encoder_percentiles, self).__init__()
        self.encoder = encoder
        self.mlp = mlp

    def encode(self, y, aux=None):
        """Encode from observed spectrum to latents

        Parameters
        ----------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra
        aux: `torch.tensor`, shape (N, n_aux)
            (optional) Batch of auxiliary inputs to MLP
        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        return self.encoder(y, aux=aux)

    def _mlp(self, s):
        """From latents to percentiles

        Parameter
        ---------
        s: `torch.tensor`, shape (N, S)
            Batch of latents

        Returns
        -------
        y_: `torch.tensor`, shape (N, 10)
            Batch of predicted percentiles
        """
        return self.mlp(s)

    def _forward(self, y,s=None):
        if s is None:
            s = self.encode(y)
        y_ = self._mlp(s)
        return s, y_

    def forward(self, y, s=None):
        """Forward method

        Transforms observed spectra into percentiles using latents (we want the percentiles)

        Parameter
        ---------
        y: `torch.tensor`, shape (N, 10)
            Batch of real percentiles
        s: `torch.tensor`, shape (N, S)
            (optional) Batch of latents. When given, encoding is omitted and these
            latents are used instead.

        Returns
        -------
        s: `torch.tensor`, shape (N, S)
            Batch of latents.
        y_: `torch.tensor`, shape (N, 10)
            Batch of predicted percentiles
        """
        s,  y_ = self._forward(y, s=s)
        return s,y_

    def loss(self, y, w=None, s=None, individual=False):
        """Weighted MSE loss

        Parameter
        --------
        y: `torch.tensor`, shape (N, 10)
            Batch of percentiles
        w: `torch.tensor`, shape (N, 10)
            Batch of weights for percentiles (optional?)
        s: `torch.tensor`, shape (N, S)
            (optional) Batch of latents. When given, encoding is omitted and these
            latents are used instead.
        individual: bool
            Whether the loss is computed for each spectrum individually or aggregated

        Returns
        -------
        float or `torch.tensor`, shape (N,) of weighted MSE loss
        """
        s,y_ = self.forward(y, s=s) #predicted percentiles
        return self._loss(y, y_)

    def _loss(self, y, y_ ):  #w not used

        # loss = total squared deviation in units of variance (MSE)

        #if w==None:
        #    loss_ind = torch.sum(0.5 * (y - y_).pow(2), dim=1) / y.shape[1]
        #else:
        #    loss_ind = torch.sum(0.5 *w* (y - y_).pow(2), dim=1) / y.shape[1]

        #L1 loss
        loss_ind = torch.sum(0.5 * torch.abs(y - y_), dim=1) / y.shape[1]

        #L2 loss
        #loss_ind = torch.sum(0.5 * (y - y_).pow(2), dim=1) / y.shape[1]

        #log cosh loss
        #loss_ind = torch.mean(torch.log(torch.cosh(y - y_ + 1e-12)))


        

        return torch.sum(loss_ind)

    @property
    def n_parameter(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class encoder_percentiles(Base_encoder_percentiles):
    """Concrete implementation of spectrum encoder

    Constructs and uses :class:`SpectrumEncoder` as encoder and :class:`MPL`.

    Parameter
    ---------
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the class:`MLP`
    act: list of callables
        Activation functions for the decoder. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    n_out: int
        Number of percentiles
    """
    def __init__(self,
                 n_latent=10,
                 n_out=9,
                 n_hidden=(16,16,16),
                 act=None,dropout_1=0,dropout_2=0
                ):

        encoder = SpectrumEncoder(None,n_latent,dropout=dropout_1)

        mlp = MLP(n_latent,n_out,n_hidden=n_hidden,act=act,dropout=dropout_2)

        super(encoder_percentiles, self).__init__(
            encoder,
            mlp,
        )
