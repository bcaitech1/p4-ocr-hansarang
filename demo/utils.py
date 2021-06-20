import torch.optim as optim
import madgrad
from networks.Attention import Attention
from networks.SATRN import SATRN
from networks.SATRN_Effnet import SATRN_Eff
from adamp import AdamP

def get_network(
    model_type,
    FLAGS,
    model_checkpoint,
    device,
    train_dataset,
):
    model = None

    if model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "CRNN":
        model = CRNN()
    elif model_type == "Attention":
        model = Attention(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "SATRN_Eff":
        model = SATRN_Eff(FLAGS, train_dataset, model_checkpoint).to(device)
    else:
        raise NotImplementedError

    return model


def get_optimizer(optimizer, params, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, momentum=0.9):
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    elif optimizer == "AdamP":
        optimizer = AdamP(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer == "MADGRAD":
        eps = 1e-06
        weight_decay = 0
        optimizer = madgrad.MADGRAD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
    else:
        raise NotImplementedError
    return optimizer

