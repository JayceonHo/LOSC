import torch
import numpy as np
import random
import shutil
from nilearn import connectome
from sklearn.model_selection import StratifiedKFold, train_test_split

def show_result(result, precisions=4):
    accuracy = result[0]
    auc_roc = result[1]
    sensitivity = result[2]
    specificity = result[3]
    print(f"Accuracy- {round(np.mean(accuracy), precisions)} Standard Deviation- {round(np.std(accuracy), precisions)}")
    print(f"AUCROC - {round(np.mean(auc_roc), precisions)}, Standard Deviation- {round(np.std(auc_roc), precisions)}")
    print(
        f"Sensitivity- {round(np.mean(sensitivity), precisions)} Standard Deviation- {round(np.std(sensitivity), precisions)}")
    print(
        f"Specificity- {round(np.mean(specificity), precisions)} Standard Deviation- {round(np.std(specificity), precisions)}")


def get_cv_index(num_samples, labels, n_folds=0, seed=2025):
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    cv_splits = []
    for train_index, test_index in skf.split(np.zeros(num_samples), labels):
        train_index, val_index = train_test_split(train_index, test_size=0.05, random_state=seed,
                                                          stratify=labels[train_index])
        cv_splits.append((train_index, val_index, test_index))
    return cv_splits


def delete_folder(path):
    try:
        shutil.rmtree(path)
        print(f"文件夹 '{path}' 已被删除。")
    except FileNotFoundError:
        print(f"错误：文件夹 '{path}' 不存在。")
    except Exception as e:
        print(f"发生错误：{e}")

def set_seed(seed_value):
    random.seed(seed_value)  # Python随机数生成器的种子
    np.random.seed(seed_value)  # Numpy随机数生成器的种子
    torch.manual_seed(seed_value)  # CPU随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)  # GPU随机数生成器的种子


def get_network(x, kind="correlation", multi_scale=1):

    # r = 0 # random.randint(0, x.shape[1] - 128)
    # x = x[:, r:r + 128, :]
    conn_measure = connectome.ConnectivityMeasure(kind=kind)
    connectivity = conn_measure.fit_transform(x)
    connectivity = np.arctanh(connectivity)
    connectivity[connectivity == np.inf] = 0
    connectivity[np.isnan(connectivity)] = 0
    connectivity[connectivity<0] = 0
    return connectivity