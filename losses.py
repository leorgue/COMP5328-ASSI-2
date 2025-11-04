# Forward Correction Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardCorrectionLoss(nn.Module):
    def __init__(self, transition_matrix):
        super(ForwardCorrectionLoss, self).__init__()
        self.transition_matrix = transition_matrix

    def forward(self, logits, noisy_labels):
        probs = F.softmax(logits, dim=1)
        corrected_probs = torch.matmul(probs, self.transition_matrix)
        corrected_probs = torch.clamp(corrected_probs, min=1e-7, max=1.0)
        loss = F.nll_loss(torch.log(corrected_probs), noisy_labels)
        return loss


# Pre-trained ResNet18 adapted for CIFAR
# Co-Teaching Loss
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    """
    Co-Teaching loss function
    y_1, y_2: logits from two networks
    t: labels
    forget_rate: percentage of data to discard
    """
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = torch.argsort(loss_1)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = torch.argsort(loss_2)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # Select small loss samples from each network
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # Exchange and update
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return loss_1_update, loss_2_update


# Backward Correction Loss
class BackwardCorrectionLoss(nn.Module):
    def __init__(self, transition_matrix):
        super(BackwardCorrectionLoss, self).__init__()
        # Use inverse of transition matrix
        self.transition_matrix_inv = torch.inverse(transition_matrix)

    def forward(self, logits, noisy_labels):
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, noisy_labels, reduction='none')

        # Get one-hot encoded noisy labels
        batch_size = noisy_labels.size(0)
        num_classes = logits.size(1)
        one_hot = torch.zeros(batch_size, num_classes).to(logits.device)
        one_hot.scatter_(1, noisy_labels.unsqueeze(1), 1)

        # Apply inverse transition matrix: T^-1 @ one_hot
        # This gives us the corrected label distribution
        corrected_labels = torch.matmul(one_hot, self.transition_matrix_inv)
        corrected_labels = torch.clamp(corrected_labels, min=0.0, max=1.0)

        # Normalize to ensure it sums to 1
        corrected_labels = corrected_labels / corrected_labels.sum(dim=1, keepdim=True)

        # Compute loss with corrected labels
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(corrected_labels * log_probs).sum(dim=1).mean()

        return loss