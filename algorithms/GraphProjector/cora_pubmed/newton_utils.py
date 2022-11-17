import torch
import math




def lr_grad(w, X, y, lam=0):
    y[y == 0] = -1

    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z - 1) * y) + lam * X.size(0) * w


def lr_hessian_inv(w, X, y, lam=0, batch_size=50000):
    y[y == 0] = -1

    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i + 1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)

    H = H + lam * X.size(0) * torch.eye(X.size(1)).float()

    try:
        H_inv = torch.linalg.inv(H)
    except:
        H_inv = torch.linalg.pinv(H)

    return H_inv