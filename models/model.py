import torch
import torch.nn as nn
import numpy as np
from models.S2CG import S2CG
from models.chebynet import ChebNet

class ClassificationModel(nn.Module):
    def __init__(self, config, cluster_label):
        super().__init__()
        self.k = config["K"]
        self.S2CG = S2CG(length=config["length"], hidden_dim=config["s2cg"]["hidden_dim"], cutoff=config["s2cg"]["cutoff"])
        self.S2CG.load_state_dict(torch.load(f"./data/{config["name"]}/s2cg.pth"))
        self.S2CG.eval()
        self.intra_cluster_learner = ChebNet(config["N"], config["model"]["hidden_dim"], config["model"]["out_dim"], k=config["model"]["order"])
        cluster_index, dims = np.unique(cluster_label, return_counts=True)
        self.ARP = nn.ModuleList()
        for i in cluster_index:
            self.ARP.append(
                nn.Sequential(
                    nn.Linear(dims[i], dims[i]),
                    nn.Sigmoid(),
                )
            )

        self.gating = nn.Sequential(
            nn.Linear(config["model"]["hidden_dim"], config["model"]["hidden_dim"]),
            nn.Sigmoid(),
        )
        norm_layer = nn.BatchNorm1d(config["model"]["mid_dim"]) if config["name"] =="hcp" else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(config["model"]["hidden_dim"]  * config["K"], config["model"]["mid_dim"]),
            norm_layer,
            nn.GELU(),
            nn.Linear(config["model"]["mid_dim"], config["model"]["num_classes"]),
        )

        self.cluster_label = cluster_label


    def forward(self, feature, time_series, adj_matrix=None):
        _, n, t = time_series.shape
        mean_intra_embed, max_intra_embed = None, None
        all_weights = None
        cluster_weights = np.zeros(self.k)
        for i in range(self.k):
            intra_ts = time_series[:, self.cluster_label==i, :]
            with torch.no_grad():
                intra_net = self.S2CG(intra_ts)[0]
            intra_feat = feature[:, self.cluster_label==i, :]
            intra_embed = self.intra_cluster_learner(intra_feat, intra_net)

            weights = self.ARP[i](intra_embed.permute(0,2,1)) # BxCXM
            weights = torch.softmax(weights, dim=-1) # BxCxM
            weights = weights.mean(1).unsqueeze(-1)
            mean_roi_embed = (intra_embed.permute(0,2,1) @  weights).squeeze(-1) # BxMxC @ Bx
            gating  = self.gating(intra_embed)
            mean_edge_embed = (gating * intra_embed).mean(1)
            intra_embed = mean_roi_embed + mean_edge_embed
            cluster_weights[i] = torch.mean(mean_edge_embed).detach().cpu().numpy()

            if i == 0:
                mean_intra_embed = intra_embed
                all_weights = weights
            else:
                mean_intra_embed = torch.cat((mean_intra_embed, intra_embed), dim=1)
                all_weights = torch.cat((all_weights, weights), dim=1)
        feat = mean_intra_embed
        feat = feat.flatten(1)
        feat = self.classifier(feat)
        gating = all_weights
        return feat, gating.squeeze(-1).cpu().detach().numpy(), cluster_weights
