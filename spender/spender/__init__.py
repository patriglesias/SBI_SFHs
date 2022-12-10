import torch
from .spender_model import MLP, SpectrumEncoder, encoder_percentiles


#function tunned by Patricia
def load_model(filename, device=None,n_latent=10,n_out=10,n_hidden=(16,16,16)):
    """Load models from file

    Parameter
    ---------
    filename: str
        Path to file which contains the torch state dictionary

    device: `torch.Device`
        Device to load model structure into

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

