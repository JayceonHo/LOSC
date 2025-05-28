import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, confusion_matrix
import networkx as nx

def cal_metrics(prediction, label):
    prediction, label = prediction.detach().cpu().numpy(), label.detach().cpu().numpy()
    probability = prediction[:, 1]
    prediction = np.argmax(prediction, axis=1)

    acc = (prediction == label).astype(np.int32).sum() / len(label)
    auc = roc_auc_score(label, probability)

    tn = np.sum((prediction == 0) & (label == 0))
    fp = np.sum((prediction == 1) & (label == 0))

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    tp = np.sum((prediction == 1) & (label == 1))
    fn = np.sum((prediction == 0) & (label == 1))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    return acc, auc, sensitivity, specificity



def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)

    # 计算每个聚类的最大类别样本数
    max_per_cluster = np.max(contingency_matrix, axis=1)

    # 计算纯度
    purity = np.sum(max_per_cluster) / np.sum(contingency_matrix)

    return purity

def cal_pco_ci(matrix, labels, K=8):
    matrix[matrix < 0] = 0
    intra_similarity, inter_similarity = 0, 0
    fco = 0
    ci = 0
    # 计算类内相似度
    cnt = 0

    for k in range(K):
        # 找出属于第k类的样本索引
        indices = np.where(labels == k)[0]
        # 计算类内相似度之和
        for i in indices:
            for j in indices:
                if i != j:
                    intra_similarity += matrix[i, j]
                    cnt += 1
    intra_similarity = intra_similarity / cnt
    # 计算类间相似度
    cnt = 0
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            # 找出属于第k1类和第k2类的样本索引
            indices_k1 = np.where(labels == k1)[0]
            indices_k2 = np.where(labels == k2)[0]
            # 计算类间相似度之和
            for i in indices_k1:
                for j in indices_k2:
                    inter_similarity += matrix[i, j]
                    cnt += 1
    inter_similarity = inter_similarity / cnt
    ci += intra_similarity / inter_similarity
    fco += intra_similarity - inter_similarity
    return ci, fco




def cal_small_world_coefficient(adj_matrix, n_random=20, seed=None):
    """
    Calculate the small-world coefficient (σ) of a graph given its adjacency matrix.

    Parameters:
    - adj_matrix: 2D numpy array (adjacency matrix of the graph)
    - n_random: Number of random graphs to generate for comparison (default: 20)
    - seed: Random seed for reproducibility (default: None)

    Returns:
    - sigma (σ): Small-world coefficient (σ >> 1 suggests small-worldness)
    - C: Clustering coefficient of the input graph
    - L: Average shortest path length of the input graph
    - C_rand: Average clustering coefficient of random graphs
    - L_rand: Average shortest path length of random graphs
    """
    # Create the graph from the adjacency matrix
    # threshold = np.percentile(adj_matrix[adj_matrix>0], 90)

    # adj_matrix = (adj_matrix>=threshold).astype(int)
    # adj_matrix[adj_matrix < 0] = 0
    adj_matrix = (adj_matrix - np.min(adj_matrix)) / (np.max(adj_matrix) - np.min(adj_matrix))

    # adj_matrix = (adj_matrix - np.mean(adj_matrix))/np.std(adj_matrix)
    # adj_matrix[adj_matrix < 0] = 0

    G = nx.from_numpy_array(adj_matrix)

    # Calculate C and L for the input graph
    C = nx.average_clustering(G, weight='weight')


    G = nx.from_numpy_array(np.max(adj_matrix) - adj_matrix)
    L = nx.average_shortest_path_length(G, weight='weight')

    return C, L

def evaluate_cluster_performance(feature_matrix, gt_label, cluster_label, K=8):
    p_sco = purity_score(gt_label, cluster_label)
    nmi = normalized_mutual_info_score(gt_label, cluster_label)
    homogeneity = homogeneity_score(gt_label, cluster_label)
    _, num_cluster = np.unique(cluster_label, return_counts=True)
    ci, fco = cal_pco_ci(feature_matrix, cluster_label, K)

    return fco, ci, p_sco, nmi, homogeneity


# Example usage:
if __name__ == "__main__":
    # Example adjacency matrix (replace with your own)
    adj_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])

    sigma, C, L, C_rand, L_rand = small_world_coefficient(adj_matrix)
    print(f"Small-world coefficient (σ): {sigma:.4f}")
    print(f"Clustering coefficient (C): {C:.4f}")
    print(f"Avg shortest path (L): {L:.4f}")
    print(f"Random graph C_rand: {C_rand:.4f}")
    print(f"Random graph L_rand: {L_rand:.4f}")
