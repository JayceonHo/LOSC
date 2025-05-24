import numpy as np
import matplotlib.pyplot as plt

def draw_cluster(cluster_result, gt_label, N=200, K=8):
    if cluster_result is None:
        cluster_result = np.load("data/abide/clusters_8.npy")
    draw = np.zeros((K, 8))

    for i in range(K):
        for j in range(N):
            cl = cluster_result[j]
            gt = gt_label[j]
            if int(cl) == i:
                draw[i, int(gt)] += 1
    clusters = list(range(1, K + 1))
    CB = draw[:, 0]
    V = draw[:, 1]
    SMN = draw[:, 2]
    DAN = draw[:, 3]
    VAN = draw[:, 4]
    L = draw[:, 5]
    FPN = draw[:, 6]
    DMN = draw[:, 7]

    bottom1 = np.array(CB)
    bottom2 = bottom1 + np.array(V)
    bottom3 = bottom2 + np.array(SMN)
    bottom4 = bottom3 + np.array(DAN)
    bottom5 = bottom4 + np.array(VAN)
    bottom6 = bottom5 + np.array(L)
    bottom7 = bottom6 + np.array(FPN)
    bottom8 = bottom7 + np.array(DMN)
    bar_width = 0.6
    fig, ax = plt.subplots()
    ax.bar(clusters, CB, width=bar_width, label='CB&SB', color='deepskyblue')
    ax.bar(clusters, V, bottom=bottom1, label='V', color='limegreen', width=bar_width)
    ax.bar(clusters, SMN, bottom=bottom2, label='SMN', color='orangered', width=bar_width)
    ax.bar(clusters, DAN, bottom=bottom3, label='DAN', color='orange', width=bar_width)
    ax.bar(clusters, VAN, bottom=bottom4, label='VAN', color='mediumpurple', width=bar_width)
    ax.bar(clusters, L, bottom=bottom5, label='L', color='dimgray', width=bar_width)
    ax.bar(clusters, FPN, bottom=bottom6, label='FPN', color='yellow', width=bar_width)
    ax.bar(clusters, DMN, bottom=bottom7, label='DMN', color='black',width=bar_width)

    ax.legend(loc='best', framealpha=0.5, ncol=2)

    plt.show()

def draw_adhd_cluster(cluster_result, gt_label, N=116, K=8):
    if cluster_result is None:
        cluster_result = np.load("data/adhd/clusters_8.npy")
    draw = np.zeros((K, 10))

    for i in range(K):
        for j in range(N):
            cl = cluster_result[j]
            gt = gt_label[j]
            if int(cl) == i:
                draw[i, int(gt)] += 1
    clusters = list(range(1, K + 1))
    CB = draw[:, 0]
    V = draw[:, 1]
    SMN = draw[:, 2]
    DAN = draw[:, 3]
    VAN = draw[:, 4]
    L = draw[:, 5]
    FPN = draw[:, 6]
    DMN = draw[:, 7]
    SN = draw[:, 8]
    AN = draw[:, 9]

    bottom1 = np.array(CB)
    bottom2 = bottom1 + np.array(V)
    bottom3 = bottom2 + np.array(SMN)
    bottom4 = bottom3 + np.array(DAN)
    bottom5 = bottom4 + np.array(VAN)
    bottom6 = bottom5 + np.array(L)
    bottom7 = bottom6 + np.array(FPN)
    bottom8 = bottom7 + np.array(DMN)
    bottom9 = bottom8 + np.array(SN)
    bottom10 = bottom9 + np.array(AN)


    bar_width = 0.5
    fig, ax = plt.subplots()
    ax.bar(clusters, CB, width=bar_width, label='CB&SB', color='deepskyblue')
    ax.bar(clusters, V, bottom=bottom1, label='V', color='limegreen', width=bar_width)
    ax.bar(clusters, SMN, bottom=bottom2, label='SMN', color='orangered', width=bar_width)
    ax.bar(clusters, DAN, bottom=bottom3, label='DAN', color='orange', width=bar_width)
    ax.bar(clusters, VAN, bottom=bottom4, label='VAN', color='mediumpurple', width=bar_width)
    ax.bar(clusters, L, bottom=bottom5, label='L', color='dimgray', width=bar_width)
    ax.bar(clusters, FPN, bottom=bottom6, label='FPN', color='yellow', width=bar_width)
    ax.bar(clusters, DMN, bottom=bottom7, label='DMN', color='black', width=bar_width)
    ax.bar(clusters, SN, bottom=bottom8, label='SN', color='red', width=bar_width)
    ax.bar(clusters, AN, bottom=bottom9, label='AN', color='blue', width=bar_width)

    ax.set_xlabel('Cluster')
    ax.set_ylabel('#ROIs in cluster')
    ax.set_xticks(clusters)
    ax.set_yticks(np.arange(10, 26, 5))
    ax.legend(loc='best', framealpha=0.5, ncol=2)

    plt.tight_layout()
    plt.show()