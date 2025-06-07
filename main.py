import random
import numpy as np
import argparse
import json
import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from models.S2CG import S2CG
from models.model import ClassificationModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import SpectralClustering
from utilis.loss import batch_rayleigh_quotient_loss
from utilis.dataset import ABIDE, HCP, ADHD200
from utilis.metric import cal_metrics, evaluate_cluster_performance, cal_small_world_coefficient
from utilis.tools import *
from utilis.plot import draw_cluster, draw_adhd_cluster

class TrainS2CG:
    def __init__(self):
        self.train_data_loader = DataLoader(train_dataset, batch_size=config["s2cg"]["batch_size"], shuffle=True)
        self.test_data_loader = DataLoader(test_dataset, batch_size=config["s2cg"]["batch_size"], shuffle=True)
        self.num_epoch = config["s2cg"]["epoch"]
        self.fco_list, self.ci_list, self.reg = [], [], []
        self.nmi_list, self.purity_list, self.homo_list = [], [], []
        self.model = S2CG(length=config["length"], hidden_dim=config["s2cg"]["hidden_dim"], cutoff=config["s2cg"]["cutoff"]).to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["s2cg"]["lr"])
        self.train()
        self.get_cluster_label()
        torch.save(self.model.state_dict(), f"./data/{args.dataset}/s2cg.pth")

    def train(self):
        for epoch in tqdm(range(1, self.num_epoch+1)):
            self.model.train()
            for i_iter, data in enumerate(self.train_data_loader):
                sim_mat, spa_sim, spec_sim = self.model(data[1])
                rayleigh_loss = config["s2cg"]["rql_loss"] * batch_rayleigh_quotient_loss(sim_mat, config["K"])
                regression_loss = config["s2cg"]["reg_loss"] * torch.mean((sim_mat - data[0]) ** 2)
                loss = rayleigh_loss + regression_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def get_cluster_label(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(train_dataset.time_series.float())
            sim_mat = out[0].cpu().numpy()
        if args.save_result:
            np.save(f"./data/{args.dataset}/fbn.npy", sim_mat)
        mat = sim_mat.copy()
        mat[mat < config["clustering_threshold"]] = 0
        mat = mat.mean(0)
        clustering_label = (SpectralClustering(n_clusters=config["K"], affinity='precomputed', n_init=1000).fit(mat)).labels_
        gt_label = np.load(config["data_root"] + "/gt_clusters.npy")
        fco, ci, p_sco, nmi, homogeneity = evaluate_cluster_performance(sim_mat.mean(0), gt_label, clustering_label, config["K"])
        np.save(f"./data/{args.dataset}/cluster.npy", clustering_label)
        if args.dataset == "adhd":
            draw_adhd_cluster(clustering_label, gt_label, config["N"], config["K"])
        else:
            draw_cluster(clustering_label, gt_label, config["N"], config["K"])
        print(f"fco {fco}, ci {ci}, purity {p_sco} , nmi: {nmi}, homo: {homogeneity}")

        swe = cal_small_world_coefficient(sim_mat.mean(0))
        swe_pc = cal_small_world_coefficient(train_dataset.feature_matrix.cpu().numpy().mean(0))
        if args.dataset == "hcp":
            swe_pcc = cal_small_world_coefficient(train_dataset.adjacency.cpu().numpy())
        else:
            swe_pcc = cal_small_world_coefficient(train_dataset.adjacency.cpu().numpy().mean(0))
        print(f"our small-worldness {swe}, pc small-worldness {swe_pc}, pcc small-worldness {swe_pcc}")


class TrainModel:
    def __init__(self):
        self.num_epoch = config["model"]["epoch"]
        cluster_label = np.load(f"./data/{args.dataset}/cluster.npy")
        self.model = ClassificationModel(config, cluster_label)
        self.model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["model"]["lr"])

        self.train_data_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
        self.test_data_loader = DataLoader(test_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
        self.train()
        self.model.load_state_dict(torch.load(f"./data/{args.dataset}/{f}_fold_best_model.pth"))
        test_result = self.evaluate()
        print(f, test_result)
        accuracy[f] = test_result[0]
        auc_roc[f] = test_result[1]
        sensitivity[f] = test_result[2]
        specificity[f] = test_result[3]

    def train(self):
        for epoch in tqdm(range(self.num_epoch)):
            train_correct = 0
            self.model.train()
            for idx, data in enumerate(self.train_data_loader):
                pred = self.model(data[0], data[1], data[-1])[0]
                loss = config["model"]["loss"] * loss_func(pred, data[2])
                train_correct += data[2].eq(torch.argmax(pred, dim=1)).sum().item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.evaluate()
            writer.add_scalar("train\tacc", train_correct/train_dataset.__len__(), epoch)
            writer.add_scalar("test\tacc", self.evaluate()[0], epoch)
        torch.save(self.model.state_dict(), f"./data/{args.dataset}/{f}_fold_best_model.pth")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            pred, rois_weights, cluster_weights =  self.model(test_dataset.feature_matrix.float(), test_dataset.time_series.float())
        if args.save_result:
            np.save(f"./data/{args.dataset}/{f}_roi_weights.npy", rois_weights)
            np.save(f"./data/{args.dataset}/{f}_cluster_weights.npy", cluster_weights)
        acc, auc, sens, spec = cal_metrics(pred, test_dataset.label)
        return acc, auc, sens, spec



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',default='configs/config.json', type=str)
    parser.add_argument('-p', '--port', default='23490', type=str)
    parser.add_argument('-d', '--dataset', default='abide', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('-s', '--save_result', action='store_true')
    parser.add_argument('-t', '--train_s2cg', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))[args.dataset]
    loss_func = torch.nn.CrossEntropyLoss(reduction=config["loss"])
    accuracy, auc_roc, sensitivity, specificity = (np.zeros(config["num_folds"]), np.zeros(config["num_folds"]),
                                                   np.zeros(config["num_folds"]), np.zeros(config["num_folds"]))
    gt_cluster = np.load(config["data_root"] + "gt_clusters.npy")
    delete_folder(f"./log/{args.dataset}_{config["atlas"]}_log/")
    for f in range(config["num_folds"]):
        writer = SummaryWriter(f"./log/{args.dataset}_{config["atlas"]}_log/{f}")
        if args.dataset == "abide":
            train_index, valid_index, test_index = (np.load(config["data_root"] + "train_index.npy"),
                                                    np.load(config["data_root"] + "valid_index.npy"),
                                                    np.load(config["data_root"] + "test_index.npy"))
            train_dataset = ABIDE(config, train_index)
            test_dataset = ABIDE(config, test_index)
        elif args.dataset == "hcp":
            train_dataset = HCP(config, f+1)
            test_dataset = HCP(config, f+1, "test")
        elif args.dataset == "adhd":
            labels = np.load(config["data_root"] + "final_label.npy")
            cv_split = get_cv_index(config["size"], labels, config["num_folds"])
            train_dataset = ADHD200(config, None, cv_split[f][0])
            test_dataset = ADHD200(config, None, cv_split[f][2])
        if args.train_s2cg and f==1:
            TrainS2CG()
        TrainModel()
        writer.close()
    show_result((accuracy, auc_roc, sensitivity, specificity))

