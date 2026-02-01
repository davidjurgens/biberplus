import torch
import torch.nn as nn
import torch.nn.functional as F


def InfoNCE_loss_full(z1, z2, temperature=0.05):
    """InfoNCE loss function"""
    loss_fn = nn.CrossEntropyLoss()
    sim = cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), temperature)
    batch_size = sim.size(0)
    I = torch.eye(batch_size).bool()  # mask out diagonal entries

    # add more in-batch negative samples
    sim = torch.concat([sim, sim.t()[~I].reshape(batch_size, batch_size - 1)], dim=1)
    sim_a = cosine_similarity(z1.unsqueeze(1), z1.unsqueeze(0), temperature)
    sim_b = cosine_similarity(z2.unsqueeze(1), z2.unsqueeze(0), temperature)

    sim = torch.concat([sim, sim_a[~I].reshape(batch_size, batch_size - 1)], dim=1)
    sim = torch.concat([sim, sim_b[~I].reshape(batch_size, batch_size - 1)], dim=1)

    labels = torch.arange(sim.size(0)).long().to(z1.device)
    loss = loss_fn(sim, labels)
    return loss


def cosine_similarity(z1, z2, temperature=0.05):
    cos = F.cosine_similarity(z1, z2, dim=-1)
    return cos / temperature
