import numpy as np
import pickle
import torch
import scanpy as sc


def save_log_file(log_list, dataset):
    file_path = f'/home/hanyuji/Results/scDYff/loss_log/loss_log_{dataset}.pt'

    with open(file_path, 'wb') as f:
        pickle.dump(log_list, f)
