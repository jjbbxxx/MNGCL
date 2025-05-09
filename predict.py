#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py

基于训练好的多模型集成，为每个基因预测是否为癌症驱动基因。
"""
import os
import argparse
import yaml
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from mngcl import MNGCL
from gcn import GCN
from tqdm import tqdm

def load_checkpoint_model(path, device, in_feats, hidden_size, out_size, tau, pos):
    """
    根据 checkpoint 构建与训练时一致的 GCN + MNGCL 模型，并加载权重。
    """
    gcn = GCN(in_feats, hidden_size, out_size).to(device)
    model = MNGCL(
        gnn=gcn,
        pos=pos.to(device),
        tau=tau,
        gnn_outsize=out_size,
        projection_hidden_size=hidden_size,
        projection_size=out_size
    ).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

@torch.no_grad()
def get_embedding(model, adj1, adj2, adj3, feats):
    """
    对单个模型前向一次，返回 emb 张量，形状 [n_genes, 3]
    """
    _, _, _, emb, _ = model(adj1, adj2, adj3, feats, feats, feats)
    return emb

def main():
    parser = argparse.ArgumentParser(description="多模型集成驱动基因预测")
    parser.add_argument('--config',      type=str, default='config.yaml', help='配置文件')
    parser.add_argument('--dataset',     type=str, default='CPDB',        help='数据集名称')
    parser.add_argument('--cancer_type', type=str, default='pan-cancer', help='癌症类型')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint 文件夹')
    parser.add_argument('--out',         type=str, default='predictions.csv', help='输出文件')
    args = parser.parse_args()

    # 1. 加载配置
    cfg_all = yaml.safe_load(open(args.config, 'r'))
    cfg = cfg_all[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载数据与图
    dataPath = os.path.join('data', args.dataset)
    data = torch.load(os.path.join(dataPath, f"{args.dataset}_data.pkl")).to(device)

    # 根据癌症类型截取特征
    if args.cancer_type == 'pan-cancer':
        data.x = data.x[:, :48]
    else:
        cancerType_dict = {
            'kirc':[0,16,32],'brca':[1,17,33],'prad':[3,19,35],'stad':[4,20,36],
            'hnsc':[5,21,37],'luad':[6,22,38],'thca':[7,23,39],'blca':[8,24,40],
            'esca':[9,25,41],'lihc':[10,26,42],'ucec':[11,27,43],'coad':[12,28,44],
            'lusc':[13,29,45],'cesc':[14,30,46],'kirp':[15,31,47]
        }
        data.x = data.x[:, cancerType_dict[args.cancer_type]]

    # 拼接 node2vec 特征
    dataz = torch.load(os.path.join(dataPath, "Str_feature.pkl"), map_location='cpu').to(device)
    data.x = torch.cat((data.x, dataz), dim=1)

    # 加载邻接矩阵
    ppiAdj   = torch.load(os.path.join(dataPath, 'ppi.pkl'), map_location='cpu')
    pathAdj  = torch.load(os.path.join(dataPath, 'pathway_SimMatrix.pkl'), map_location='cpu')
    goAdj    = torch.load(os.path.join(dataPath, 'GO_SimMatrix.pkl'),      map_location='cpu')
    ppi_self = torch.load(os.path.join(dataPath, 'ppi_selfloop.pkl'),     map_location='cpu')
    pos = ppi_self.to_dense()

    # 将邻接索引和特征提前搬到设备
    adj1 = ppiAdj.coalesce().indices().to(device)
    adj2 = pathAdj.coalesce().indices().to(device)
    adj3 = goAdj.coalesce().indices().to(device)
    feats = data.x.to(device)

    # 3. 确定模型维度参数
    in_feats = data.x.shape[1]
    if args.cancer_type == 'pan-cancer':
        hidden_size = 300  # 与训练脚本中 GCN(64,300,100) 保持一致
        out_size    = 100
    else:
        hidden_size = 150  # 与训练脚本中 GCN(19,150,50) 保持一致
        out_size    = 50
    tau = cfg['tau']

    # 4. 收集 checkpoint 路径
    ckpts = sorted([
        os.path.join(args.checkpoint_dir, fn)
        for fn in os.listdir(args.checkpoint_dir) if fn.endswith('.pth')
    ])

    # 5. 无监督获取每个模型的 emb
    emb_list = []
    for ckpt in tqdm(ckpts, desc='Embedding'):
        model = load_checkpoint_model(ckpt, device, in_feats, hidden_size, out_size, tau, pos)
        emb = get_embedding(model, adj1, adj2, adj3, feats)
        emb = torch.sigmoid(emb).cpu().numpy()
        emb_list.append(emb)

    # 6. 集成：对所有 emb 在第 0 维取平均
    emb_avg = np.stack(emb_list, axis=0).mean(axis=0)  # [n_genes, 3]

    # 7. 使用已知标签基因训练 LogisticRegression
    # 使用 numpy 进行逻辑或，兼容 data.mask 可能为 numpy.ndarray
    mask_train = np.logical_or(data.mask, data.mask_te)
    # 同样用 numpy 合并标签并转换为整数
    y_train    = np.logical_or(data.y, data.y_te).astype(int)
    clf = LogisticRegression(max_iter=10000)
    clf.fit(emb_avg[mask_train], y_train[mask_train])

    # 8. 对所有基因预测概率并写入 CSV
    probs = clf.predict_proba(emb_avg)[:, 1]
    with open(args.out, 'w') as f:
        f.write('gene_index,probability\n')
        for idx, p in enumerate(probs):
            f.write(f"{idx},{p:.6f}\n")

    print(f"预测完成，结果保存在 {args.out}")

if __name__ == '__main__':
    main()