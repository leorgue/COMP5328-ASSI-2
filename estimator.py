import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

@torch.no_grad()
def _collect_probs(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1).detach().cpu()
        probs.append(p)
    return torch.cat(probs, dim=0)  # N by K

@torch.no_grad()
def initial_t_via_pseudo_anchors(model, train_loader, num_classes, device, top_k=1):
    """
    Initial T estimate using high confidence pseudo anchors per class.
    """
    P = _collect_probs(model, train_loader, device)        # N by K predicted noisy posteriors
    K = num_classes
    T_init = torch.zeros(K, K, dtype=torch.float32)
    for i in range(K):
        idx = torch.topk(P[:, i], k=min(top_k, P.shape[0])).indices
        row = P[idx].mean(dim=0)
        row = torch.clamp(row, min=1e-8)
        row = row / row.sum()
        T_init[i] = row
    return T_init.to(device)

def revise_T_with_slack(model, train_loader, T_init, epochs=5, lr=1e-3, device="cuda"):
    """
    Learn a small slack on top of T_init by minimizing reweighted NLL on noisy labels.
    T is row stochastic after each step.
    """
    model.eval()
    T_hat = T_init.clone().detach().to(device)
    T_hat.requires_grad = True
    opt = torch.optim.Adam([T_hat], lr=lr)

    for _ in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            g = F.softmax(model(x), dim=1)       # estimate of clean posterior
            p_noisy = torch.matmul(g, T_hat.t()) # induced noisy posterior

            # importance weights for risk consistent correction
            num = g.gather(1, y.view(-1, 1)).squeeze()
            den = p_noisy.gather(1, y.view(-1, 1)).squeeze() + 1e-8
            w = num / den

            loss_vec = F.nll_loss(torch.log(p_noisy + 1e-8), y, reduction="none")
            loss = (w * loss_vec).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        # row wise projection to probability simplex
        with torch.no_grad():
            T_hat.data = torch.clamp(T_hat.data, min=0.0)
            row_sums = T_hat.data.sum(dim=1, keepdim=True) + 1e-12
            T_hat.data = T_hat.data / row_sums

    return T_hat.detach()

def estimate_transition_matrix(model, train_dataset, batch_size, num_workers, num_classes,
                               device, top_k=1, revise_epochs=5, revise_lr=1e-3):
    """
    One call helper used from train.py
    Returns T_est as torch FloatTensor on device
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    T_init = initial_t_via_pseudo_anchors(model, train_loader, num_classes, device, top_k=top_k)
    T_est = revise_T_with_slack(model, train_loader, T_init, epochs=revise_epochs,
                                lr=revise_lr, device=device)
    return T_est
