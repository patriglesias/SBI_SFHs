import torch
from .spender_model import MLP, SpectrumEncoder, encoder_percentiles


def load_model(filename,n_latent=16,n_out=9,n_hidden=(16,16,16)):
    """
    Load encoder models from file

    Parameter
    ---------
    filename: str
        Path to file which contains the torch state dictionary

    n_latent: int
        Number of components for the latent vectors
    
    n_out: int
        Number of features to predict

    n_hidden: tuple
        A tuple of size equal to the number of hidden layers, and with values equal to the number of units
        in each layer

    Returns
    -------
    model: `torch.nn.Module`
        The default :class`encoder_percentiles` model loaded from file
    loss: `torch.tensor`
        Traning and validation loss for this model
    """

    model_struct = torch.load(filename, map_location=device)

    model = encoder_percentiles(n_latent=n_latent,n_out=n_out,n_hidden=n_hidden)

    model.load_state_dict(model_struct['model'], strict=False)

    loss = torch.tensor(model_struct['losses'])
    return model, loss

