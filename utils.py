import numpy as np
import pickle
import torch
import scanpy as sc
import geomloss


# 保存训练损失
def save_log_file(log_list, dataset):
    file_path = f'/home/hanyuji/Results/scDYff/loss_log/loss_log_{dataset}.pt'

    with open(file_path, 'wb') as f:
        pickle.dump(log_list, f)


# 保存模型参数
def save_model(model, dataset_name, epochs):
    model_path = f'/home/hanyuji/Results/scDYff/{dataset_name}/model_inter_{dataset_name}_{epochs}epochs.pt'
    torch.save(model.state_dict(), model_path)


# 加载模型
# model.load_state_dict(torch.load(model_path))


# 损失函数 OT
def SinkhornLoss(t_true, t_est, blur=0.05, scaling=0.5):
    ot_solver = geomloss.SamplesLoss(
        "sinkhorn", p=2, blur=blur, scaling=scaling, debias=True, backend="tensorized"
    )
    loss = ot_solver(t_true, t_est)

    return loss
