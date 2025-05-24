import torch

def batch_rayleigh_quotient_loss(A, K=8, mode="mean"):
    B, N, _ = A.shape

    # D = torch.diag_embed(A.sum(dim=2))  # (B, N, N)
    D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2) + 1e-8))  # 避免除零
    # print("D shape: ", D_inv_sqrt.shape)
    L = torch.eye(A.size(1), device=A.device) - torch.bmm(D_inv_sqrt, torch.bmm(A, D_inv_sqrt))

    if mode == "mean":
        L = torch.mean(L, dim=0)
        if torch.isfinite(L).all():
            eigenvalue = torch.linalg.eigvals(L)
            values, _ = torch.topk(torch.abs(eigenvalue), K, largest=False)
            loss = torch.sum(values)
        else:
            loss = torch.tensor(0., device=A.device)
            print("Invalid L")
    else:
        for i in range(L.shape[0]):
            loss, cnt = 0, 0
            if torch.isfinite(L[i]).all():
                eigenvalue = torch.linalg.eigvals(L[i])
                values, _ = torch.topk(torch.abs(eigenvalue), K, largest=False)
                loss += torch.sum(values)
                cnt += 1
            else:
                continue
            loss = loss / cnt
    return loss