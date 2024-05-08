import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCSE(nn.Module):
    def __init__(self, temperature=0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, sample, sample_aug):  # shape: (N, d)
        batch_size, _ = sample.size()

        sim_matrix = F.cosine_similarity(sample.unsqueeze(dim=1), sample_aug.unsqueeze(0) , dim=-1)
        sim_matrix = sim_matrix / self.temperature

        # labels := [0, 1, 2, ..., batch_size - 1]
        # labels indicate the index of the diagonal element (i.e. positive examples)
        labels = torch.arange(batch_size).long().to(sample.device)
        # it may seem strange to use Cross-Entropy Loss here.
        # this is a shorthund of doing SoftMax and maximizing the similarity of diagonal elements
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


if __name__ == "__main__":
    hidden = torch.randn(32, 128)
    hidden_aug = torch.randn(32, 128)

    criterion = SimCSE()
    print(criterion(hidden, hidden_aug))
