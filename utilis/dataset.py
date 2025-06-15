import torch
import numpy as np
from utilis.tools import get_network


def normalize(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (x - mean) / std

class ABIDE(torch.utils.data.Dataset):
    def __init__(self, config, index):
        super().__init__()
        self.root = config["data_root"]
        data = np.load(self.root + "abide.npy", allow_pickle=True).tolist()
        del data["site"] # we use all sites data
        data = map(lambda x: torch.from_numpy(x[index]).to(config["device"]), data.values())
        self.time_series, self.label, self.feature_matrix, self.adjacency = data

    def __len__(self):
        return self.feature_matrix.shape[0]

    def __getitem__(self, idx):
        return self.feature_matrix[idx].float(), self.time_series[idx].float(), self.label[idx].long(), self.adjacency[idx].float()


class HCP(torch.utils.data.Dataset):
    def __init__(self, config, fold, mode="train"):
        super().__init__()
        self.root = config["data_root"]
        ts, label, adj = np.load(self.root+f"{mode}_data_{fold}.npy"), np.load(self.root+f"{mode}_label_{fold}.npy"), np.load(self.root+f"adj_matrix.npy")
        ts = ts.squeeze(1).squeeze(-1)
        feat = get_network(ts)
        ts = ts.transpose(0,2,1)
        data = (ts, label, feat, adj)
        data = map(lambda x: torch.from_numpy(x).to(config["device"]), data)
        self.time_series, self.label, self.feature_matrix, self.adjacency = data

    def __len__(self):
        return self.feature_matrix.shape[0]

    def __getitem__(self, idx):
        return self.feature_matrix[idx].float(), self.time_series[idx].float(), self.label[idx].long(), self.adjacency.float()


class ADHD200(torch.utils.data.Dataset):
    def __init__(self, config, site, split):
        super().__init__()
        self.root = config["data_root"]
        feat, ts, label = (np.load(self.root + "final_matrix.npy"), np.load(self.root+"final_ts.npy"),
                                np.load(self.root+"final_label.npy"))
        label[label>1] = 1
        adj = get_network(ts, "partial correlation")
        selected_index = np.load(self.root+"selected_index.npy")
        feat, ts, label, adj = feat[selected_index], ts[selected_index], label[selected_index], adj[selected_index]
        feat, ts, label, adj = feat[split], ts[split].transpose(0,2,1), label[split], adj[split]
        data = (ts, label, feat, adj)
        data = map(lambda x: torch.from_numpy(x).to(config["device"]), data)
        self.time_series, self.label, self.feature_matrix, self.adjacency = data

    def __len__(self):
        return self.feature_matrix.shape[0]

    def __getitem__(self, idx):
        return self.feature_matrix[idx].float(), self.time_series[idx].float(), self.label[idx].long(), self.adjacency.float()


