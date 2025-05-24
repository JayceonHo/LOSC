import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, confusion_matrix

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

def evaluate_cluster_performance(feature_matrix, gt_label, cluster_label, K=8):
    p_sco = purity_score(gt_label, cluster_label)
    nmi = normalized_mutual_info_score(gt_label, cluster_label)
    homogeneity = homogeneity_score(gt_label, cluster_label)
    _, num_cluster = np.unique(cluster_label, return_counts=True)
    ci, fco = cal_pco_ci(feature_matrix, cluster_label, K)

    return fco, ci, p_sco, nmi, homogeneity
