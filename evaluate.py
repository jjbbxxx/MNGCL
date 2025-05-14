# evaluate.py

import os
import argparse
import yaml
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from predict import load_checkpoint_model, get_embedding  # 复用 predict.py 中的函数

def evaluate(args):
    # 1. 读取配置
    cfg_all = yaml.safe_load(open(args.config, 'r'))
    cfg = cfg_all[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载数据
    data_path = os.path.join('data', args.dataset)
    data = torch.load(os.path.join(data_path, f"{args.dataset}_data.pkl")).to(device)
    # 拼接特征同 predict.py 中逻辑
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
    # 加载并拼接 node2vec 特征
    dataz = torch.load(os.path.join(data_path, "Str_feature.pkl"), map_location='cpu').to(device)
    data.x = torch.cat((data.x, dataz), dim=1)

    # 3. 加载邻接矩阵
    ppiAdj   = torch.load(os.path.join(data_path, 'ppi.pkl'), map_location='cpu')
    pathAdj  = torch.load(os.path.join(data_path, 'pathway_SimMatrix.pkl'), map_location='cpu')
    goAdj    = torch.load(os.path.join(data_path, 'GO_SimMatrix.pkl'),      map_location='cpu')
    ppi_self = torch.load(os.path.join(data_path, 'ppi_selfloop.pkl'),     map_location='cpu')
    pos = ppi_self.to_dense()
    adj1 = ppiAdj.coalesce().indices().to(device)
    adj2 = pathAdj.coalesce().indices().to(device)
    adj3 = goAdj.coalesce().indices().to(device)
    feats = data.x.to(device)

    # 4. 模型维度
    in_feats = data.x.shape[1]
    if args.cancer_type == 'pan-cancer':
        hidden_size, out_size = 300, 100
    else:
        hidden_size, out_size = 150, 50
    tau = cfg['tau']

    # 5. 收集所有 checkpoint，计算平均 embedding
    ckpt_files = sorted([
        os.path.join(args.checkpoint_dir, f)
        for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth')
    ])
    emb_list = []
    for ckpt in ckpt_files:
        model = load_checkpoint_model(ckpt, device, in_feats, hidden_size, out_size, tau, pos)
        emb = get_embedding(model, adj1, adj2, adj3, feats)      # 返回 [n_genes, 3]
        emb = torch.sigmoid(emb).cpu().numpy()
        emb_list.append(emb)
    emb_avg = np.stack(emb_list, axis=0).mean(axis=0)  # [n_genes, 3]

    # 6. 划分训练/测试
    # Handle possible numpy arrays or torch tensors
    mask_train = data.mask.astype(bool) if isinstance(data.mask, np.ndarray) else data.mask.cpu().numpy().astype(bool)
    y_train    = data.y.astype(int)     if isinstance(data.y, np.ndarray)    else data.y.cpu().numpy().astype(int)
    mask_test  = data.mask_te.astype(bool) if isinstance(data.mask_te, np.ndarray) else data.mask_te.cpu().numpy().astype(bool)
    y_test     = data.y_te.astype(int)     if isinstance(data.y_te, np.ndarray)    else data.y_te.cpu().numpy().astype(int)

    # 7. 训练逻辑回归
    clf = LogisticRegression(max_iter=10000)
    clf.fit(emb_avg[mask_train], y_train[mask_train])

    # 8. 在测试集上预测并计算指标
    probs_test = clf.predict_proba(emb_avg[mask_test])[:, 1]
    preds_test = (probs_test >= 0.5).astype(int)

    acc    = accuracy_score(y_test[mask_test], preds_test)
    prec   = precision_score(y_test[mask_test], preds_test)
    rec    = recall_score(y_test[mask_test], preds_test)
    f1     = f1_score(y_test[mask_test], preds_test)
    rocauc = roc_auc_score(y_test[mask_test], probs_test)
    prauc  = average_precision_score(y_test[mask_test], probs_test)
    cm     = confusion_matrix(y_test[mask_test], preds_test)

    # 9. 打印结果
    print("===== Test Set Evaluation =====")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"ROC AUC     : {rocauc:.4f}")
    print(f"PR AUC      : {prauc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MNGCL on test set")
    parser.add_argument('--config',          type=str, default='config.yaml')
    parser.add_argument('--dataset',         type=str, default='CPDB')
    parser.add_argument('--cancer_type',     type=str, default='pan-cancer')
    parser.add_argument('--checkpoint_dir',  type=str, default='checkpoints')
    args = parser.parse_args()
    evaluate(args)