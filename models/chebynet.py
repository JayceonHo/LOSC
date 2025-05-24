import torch.nn as nn
import torch

def chebyshev_polynomial(L, K):
    """
    计算切比雪夫多项式（支持批处理）
    :param L: 归一化的拉普拉斯矩阵 (B, N, N)
    :param K: 切比雪夫多项式的阶数
    :return: 切比雪夫多项式的列表 [T_0, T_1, ..., T_K]，每个元素的形状为 (B, N, N)
    """
    B, N, _ = L.shape
    T_k = []
    T_k.append(torch.eye(N).unsqueeze(0).repeat(B, 1, 1).to(L.device))  # T_0 = I (B, N, N)
    T_k.append(L)  # T_1 = L (B, N, N)

    for k in range(2, K+1):
        T_k.append(2 * torch.bmm(L, T_k[-1]) - T_k[-2])  # T_k = 2 * L * T_{k-1} - T_{k-2}
    return T_k

class ChebNetLayer(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        """
        :param in_features: 输入特征维度
        :param out_features: 输出特征维度
        :param K: 切比雪夫多项式的阶数
        """
        super(ChebNetLayer, self).__init__()
        self.K = k
        self.weights = nn.Parameter(torch.FloatTensor(k+1, in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_normal_(self.weights)
    def forward(self, x, L):
        """
        :param x: 输入特征矩阵 (B, N, in_features)
        :param L: 归一化的拉普拉斯矩阵 (B, N, N)
        :return: 输出特征矩阵 (B, N, out_features)
        """
        T_k = chebyshev_polynomial(L, self.K)  # 计算切比雪夫多项式，返回列表 [T_0, T_1, ..., T_K]，每个元素的形状为 (B, N, N)
        x_tilde = []
        for k in range(self.K+1):
            x_tilde.append(torch.bmm(T_k[k], x))  # T_k(L) * x，形状为 (B, N, in_features)
        x_tilde = torch.stack(x_tilde, dim=1)  # (B, K+1, N, in_features)
        x_tilde = torch.einsum('bkni,kio->bno', x_tilde, self.weights)  # 加权求和，形状为 (B, N, out_features)
        return x_tilde

def normalize_laplacian(A, normalize=False):
    """
    归一化拉普拉斯矩阵
    :param A: 邻接矩阵 (B, N, N)
    :return: 归一化的拉普拉斯矩阵 (B, N, N)
    """
    # k = int(80 / 100 * torch.numel(A))
    # threshold, _ = torch.kthvalue(torch.flatten(A), k)
    # A[A <= threshold] = 0

    D =torch.sum(A, dim=1)
    # D = D + 1e-4
    D = torch.diag_embed(D)
    if normalize is True:
        D_inv_sqrt = torch.inverse(torch.sqrt(D))
        I = torch.eye(A.shape[1]).unsqueeze(0).repeat(A.shape[0], 1, 1).to(A.device)  # 单位矩阵 (B, N, N)
        return I - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)  # L = I - D^{-1/2} A D^{-1/2}
    else:
        return D - A


class ChebNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, normalize=False, k=1):
        """
        :param in_features: 输入特征维度
        :param hidden_features: 隐藏层特征维度
        :param out_features: 输出特征维度
        :param K: 切比雪夫多项式的阶数
        """
        super(ChebNet, self).__init__()

        self.layer1 = ChebNetLayer(in_features, hidden_features, k=k)

        if out_features != "None":
            self.layer2 = ChebNetLayer(hidden_features, out_features, k=k)
        else:
            self.layer2 = nn.Identity()
        self.out_features = out_features
        self.normalize = normalize
        self.activation_layer = nn.GELU() # nn.LeakyReLU(0.2)
    def forward(self, x, adj_matrix):
        L = normalize_laplacian(adj_matrix)
        x = self.activation_layer(self.layer1(x, L))
        if self.out_features == "None":
            return self.layer2(x)
        else:
            return self.layer2(x, L)
