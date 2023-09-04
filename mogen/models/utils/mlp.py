import torch.nn as nn


def build_MLP(dim_list, latent_dim):
    model_list = []
    prev = dim_list[0]
    for cur in dim_list[1:]:
        model_list.append(nn.Linear(prev, cur))
        model_list.append(nn.GELU())
        prev = cur
    model_list.append(nn.Linear(prev, latent_dim))
    model = nn.Sequential(*model_list)
    return model
