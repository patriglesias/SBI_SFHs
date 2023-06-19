import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
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

    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the :class:`MLP`
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """
    def __init__(self,
                 n_latent,
                 n_hidden=(128, 64, 32),
                 act=None,
                 dropout=0):

        super(SpectrumEncoder, self).__init__()
        self.n_latent = n_latent

        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1] // 2

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s, padding=s//2) for s in sizes[:2])
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features to latents
        if act is None:
            act = [ nn.PReLU(n) for n in n_hidden ]
            # last activation identity to have latents centered around 0
            act.append(nn.Identity())
        self.mlp = MLP(self.n_feature, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout)


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
            #convs.append(nn.Sequential(conv, act, drop))
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

    def forward(self, x):
        """Forward method

        Parameters
        ----------
        x: `torch.tensor`, shape (N, 4300)
            Batch of observed spectra

        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        # run through CNNs
        h, a = self._downsample(x)
        
        # softmax attention
        a = self.softmax(a)
             
        # apply attention
        s = torch.sum(h * a, dim=2)

        #final MLP
        s = self.mlp(s)
        return s


#generate latents optimized for getting percentiles

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

    def encode(self, y):
        """Encode from observed spectrum to latents

        Parameters
        ----------
        y: `torch.tensor`, shape (N, 4300)
            Batch of observed spectra

        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        return self.encoder(y)

    def _mlp(self, s):
        """From latents to percentiles

        Parameter
        ---------
        s: `torch.tensor`, shape (N, S)
            Batch of latents

        Returns
        -------
        y_: `torch.tensor`, shape (N, 9)
            Batch of predicted percentiles
        """
        return self.mlp(s)

    def _forward(self, x):
        """Apply the two steps: encoding and MLP

        Parameter
        ---------
        x: `torch.tensor`, shape (N, 4300)
            Batch of spectra

        Returns
        -------
        s: `torch.tensor`, shape (N, S)
            Batch of latents
        y_: `torch.tensor`, shape (N, 9)
            Batch of predicted percentiles
        """

        s = self.encode(x)
        y_ = self._mlp(s)
        return s, y_

    def forward(self, x):
        """Forward method

        Transforms observed spectra into percentiles using latents (we want the percentiles)

        Parameter
        ---------
        x: `torch.tensor`, shape (N, 4300)
            Batch of spectra

        Returns
        -------
        s: `torch.tensor`, shape (N, S)
            Batch of latents.
        y_: `torch.tensor`, shape (N, 9)
            Batch of predicted percentiles
        """
        s,  y_ = self._forward(x)
        return s,y_

    def loss(self,x,y):
        """Compute loss

        Parameter
        --------
        x:  `torch.tensor`, shape (N, 4300)
            Batch of percentiles
        y: `torch.tensor`, shape (N, 9)
            Batch of percentiles
        
        Returns
        -------
        float of loss
        """
        s,y_ = self.forward(x) #give batch of spectra, return batch of latent vectors and predicted percentiles
        return self._loss(y, y_) #compute loss with predicted vs real percentiles

    def _loss(self, y, y_ ):  

        #log cosh loss
        loss_ind = torch.mean(torch.log(torch.cosh(y - y_ + 1e-12)))
        return torch.sum(loss_ind)


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

        #instrument by the moment is always None
        encoder = SpectrumEncoder(n_latent,dropout=dropout_1)

        mlp = MLP(n_latent,n_out,n_hidden=n_hidden,act=act,dropout=dropout_2)

        super(encoder_percentiles, self).__init__(
            encoder,
            mlp,
        )
