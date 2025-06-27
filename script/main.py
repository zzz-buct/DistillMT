import os.path as osp
import time
import torch
import os
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, roc_auc_score, f1_score
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Dataset, HeteroData
from torch_scatter import scatter_max
from GNN_models.base_model import STnet, Tenet
from MyDataset import MyDataset
from MyHeteroDataset import MyHeteroDataset
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore")


def get_node_features(data):
    if isinstance(data, HeteroData):
        return data['node'].x
    return data.x


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--project', type=str,
                        help='name of project')
    parser.add_argument('--train_dataset', type=str,
                        help='name of train_dataset')
    parser.add_argument('--valid_dataset', type=str,
                        help='name of valid_dataset')
    parser.add_argument('--test_dataset', type=str,
                        help='name of test_dataset')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden feature_map dim (default: 256)')
    parser.add_argument('--nlayers', type=int, default=4,
                        help='gnn layer numbers (default: 4)')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='gat heads num (default: 4)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--with_bn', type=bool, default=True,
                        help='if with bn (default: True)')
    parser.add_argument('--with_bias', type=bool, default=True,
                        help='if with bias (default: True)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight decay of optimizer (default: 5e-5)')
    parser.add_argument('--scheduler_patience', type=int, default=50,
                        help='scheduler patience (default: 50)')
    parser.add_argument('--scheduler_factor', type=float, default=0.1,
                        help='scheduler factor (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='the weight of distill loss(default: 0.1)')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for consistency loss')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='linear attention temprature(default: 0.1)')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='early stoping epoches (default:50)')
    parser.add_argument('--train_mode', type=str, default="T",
                        help='train mode T,S')
    parser.add_argument('--checkpoints_path', type=str, default="checkpoints",
                        help='teacher model save file path')
    parser.add_argument('--result_path', type=str, default="results",
                        help='three type models results save path')
    parser.add_argument('--backbone', type=str, default="HGT",
                        help='backbone models: GAT, GCN, GIN, SAGE, HGT')
    parser.add_argument('--runs', type=int, default=1, help='ten-fold cross validation')
    parser.add_argument('--early_stop_metric', type=str, default='val_loss',
                        choices=['val_loss', 'auc', 'mcc', 'f1', 'ba', 'acc'],
                        help='Metric used for early stopping and saving best model')
    parser.add_argument('--use_graph_class_weights', action='store_true', default=False,
                        help='Use class weights for graph classification loss (default=False)')
    parser.add_argument('--use_node_class_weights', action='store_true', default=False,
                        help='Use class weights for node classification loss (default=False)')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoints_path = f'{args.checkpoints_path}/{args.backbone}'

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)

    if 'HGT' in args.backbone:
        path = osp.join(osp.dirname(osp.realpath(__file__)), './HeteroData', args.project)
        train_dataset = MyHeteroDataset(path, name=args.train_dataset, use_node_attr=True, use_edge_attr=True,
                                        force_reload=False).shuffle()
        valid_dataset = MyHeteroDataset(path, name=args.valid_dataset, use_node_attr=True, use_edge_attr=True,
                                        force_reload=False)
        test_dataset = MyHeteroDataset(path, name=args.test_dataset, use_node_attr=True, use_edge_attr=True,
                                       force_reload=False)
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), './data', args.project)
        train_dataset = MyDataset(path, name=args.train_dataset, use_node_attr=True, use_edge_attr=True,
                                  force_reload=True).shuffle()
        valid_dataset = MyDataset(path, name=args.valid_dataset, use_node_attr=True, use_edge_attr=True,
                                  force_reload=True)
        test_dataset = MyDataset(path, name=args.test_dataset, use_node_attr=True, use_edge_attr=True,
                                 force_reload=True)

    x_feat = get_node_features(train_dataset[0])
    if x_feat is None:
        raise ValueError(
            "Missing node features in train_dataset, please process them as non-empty features before training!")
    input_dim = x_feat.size(1)

    num_classes = len(set([data.y.item() for data in train_dataset]))

    train_labels = [data.y.item() for data in train_dataset]
    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    if args.train_mode in ['S', 'O']:
        model = STnet(nfeat=input_dim, nhid=args.nhid, nclass=num_classes, gnn=args.backbone, nlayers=args.nlayers,
                      gat_heads=args.gat_heads, dropout=args.dropout, with_bn=args.with_bn,
                      with_bias=args.with_bias).to(device)
    else:
        model = Tenet(nfeat=input_dim, nhid=args.nhid, nclass=num_classes, gnn=args.backbone, nlayers=args.nlayers,
                      gat_heads=args.gat_heads, dropout=args.dropout, tau=args.tau, with_bn=args.with_bn,
                      with_bias=args.with_bias).to(device)

    if args.early_stop_metric == 'val_loss':
        best_metric = float('inf')
    else:
        best_metric = 0

    node_labels_all = []
    for data in train_dataset:
        if isinstance(data, HeteroData):
            node_labels_all.extend(data["node"].y.tolist())
        else:
            node_labels_all.extend(data.y_node.tolist())
    node_labels_all = [y for y in node_labels_all if y != -1]

    node_class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(node_labels_all),
                                                 y=node_labels_all)
    node_class_weights = torch.tensor(node_class_weights_np, dtype=torch.float32).to(device)

    if args.use_node_class_weights:
        ce_node_loss = torch.nn.CrossEntropyLoss(weight=node_class_weights)
    else:
        ce_node_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=args.scheduler_patience,
                                                           factor=args.scheduler_factor,
                                                           verbose=True)

    if args.use_graph_class_weights:
        ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    test_acc_all = []
    auc_all, f1_all, ba_all, mcc_all = [], [], [], []

    if args.train_mode == 'S':
        num_k = args.runs
    else:
        num_k = 1

    for idd in range(num_k):
        print("========================= Run {} / {} =========================".format(idd + 1, num_k))

        train_loader = DataLoader(train_dataset.shuffle(), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_test_acc = 0.0
        wait = 0
        best_test_acc, best_auc, best_f1 = 0.0, 0.0, 0.0

        for epoch in range(args.epochs):
            s_time = time.time()
            train_loss = 0.
            train_corrects = 0
            model.train()

            if args.train_mode == 'S':
                teacher_model = torch.load(f'{checkpoints_path}/{args.project}_teacher.pth').to(device)
                teacher_model.eval()

            for i, data in enumerate(train_loader):
                s = time.time()
                data = data.to(device)
                optimizer.zero_grad()

                if isinstance(data, HeteroData):
                    batch_vector = data['node'].batch
                    inds = torch.bincount(batch_vector).to(device)
                else:
                    inds = torch.tensor([data.ptr[i + 1] - data.ptr[i] for i in range(data.y.shape[0])]).to(device)

                labs = torch.repeat_interleave(data.y, inds)

                if args.train_mode == 'S':
                    if isinstance(data, HeteroData):
                        x = data['node'].x
                        edge_index = data['node', 'edge', 'node'].edge_index
                        batch = data['node'].batch
                    else:
                        x = data.x
                        edge_index = data.edge_index
                        batch = data.batch
                    graph_out, node_out, st_graph_embed = model(x, edge_index, batch)
                    _, te_graph_embed = teacher_model(x, labs, edge_index, batch)
                    loss_distill = mse_loss(st_graph_embed, te_graph_embed)
                    loss_classification = ce_loss(graph_out, data.y.view(-1))
                    loss = loss_classification + args.alpha * loss_distill
                    graph_labels = data.y.view(-1)
                    if args.backbone == 'HGT':
                        node_labels = data['node'].y.view(-1)
                    else:
                        node_labels = data.y_node.view(-1)
                    loss_graph = ce_loss(graph_out, graph_labels)

                    node_mask = node_labels != -1
                    if node_mask.sum() > 0:
                        loss_node = ce_node_loss(node_out[node_mask], node_labels[node_mask])
                    else:
                        soft_label = torch.full_like(node_labels, 0.5).float()
                        soft_label[graph_labels == 0] = 0.0
                        node_probs = torch.softmax(node_out, dim=1)[:, 1]
                        loss_node = F.mse_loss(node_probs, soft_label.float())

                    graph_prob = torch.softmax(graph_out, dim=1)[:, 1]
                    node_prob = torch.softmax(node_out, dim=1)[:, 1]
                    node_max_prob, _ = scatter_max(node_prob, batch, dim=0)
                    consistency = F.mse_loss(node_max_prob, graph_prob)

                    loss_distill = mse_loss(st_graph_embed, te_graph_embed)

                    loss = loss_graph + loss_node + args.alpha * loss_distill + args.beta * consistency


                else:
                    if isinstance(data, HeteroData):
                        x = data['node'].x
                        edge_index = data['node', 'edge', 'node'].edge_index
                        batch = data['node'].batch
                    else:
                        x = data.x
                        edge_index = data.edge_index
                        batch = data.batch

                    graph_out, _ = model(x, labs, edge_index, batch)

                    loss_classification = ce_loss(graph_out, data.y.view(-1))
                    loss = loss_classification

                loss.backward()
                train_loss += loss.item()
                train_corrects += graph_out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
                optimizer.step()

            train_loss /= len(train_loader)
            train_acc = train_corrects / len(train_dataset)
            scheduler.step(train_loss)

            val_loss = 0.
            val_corrects = 0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    data = data.to(device)

                    if isinstance(data, HeteroData):
                        batch_vector = data['node'].batch
                        inds = torch.bincount(batch_vector).to(device)
                    else:
                        inds = torch.tensor([data.ptr[i + 1] - data.ptr[i] for i in range(data.y.shape[0])]).to(device)
                    labs = torch.repeat_interleave(data.y, inds)
                    if args.train_mode == 'S':
                        if args.backbone == 'HGT':
                            x = data['node'].x
                            edge_index = data['node', 'edge', 'node'].edge_index
                            batch = data['node'].batch
                        else:
                            x = data.x
                            edge_index = data.edge_index
                            batch = data.batch
                        graph_out, node_out, st_graph_embed = model(x, edge_index, batch)
                        _, te_graph_embed = teacher_model(x, labs, edge_index, batch)
                        loss_distill = mse_loss(st_graph_embed, te_graph_embed)
                        loss_classification = ce_loss(graph_out, data.y.view(-1))
                        loss = loss_classification + args.alpha * loss_distill
                        graph_labels = data.y.view(-1)
                        if args.backbone == 'HGT':
                            node_labels = data['node'].y.view(-1)
                        else:
                            node_labels = data.y_node.view(-1)

                        loss_graph = ce_loss(graph_out, graph_labels)

                        node_mask = node_labels != -1
                        if node_mask.sum() > 0:
                            loss_node = ce_node_loss(node_out[node_mask], node_labels[node_mask])
                        else:
                            soft_label = torch.full_like(node_labels, 0.5).float()
                            soft_label[graph_labels == 0] = 0.0
                            node_probs = torch.softmax(node_out, dim=1)[:, 1]
                            loss_node = F.mse_loss(node_probs, soft_label.float())

                        graph_prob = torch.softmax(graph_out, dim=1)[:, 1]
                        node_prob = torch.softmax(node_out, dim=1)[:, 1]
                        node_max_prob, _ = scatter_max(node_prob, batch, dim=0)
                        consistency = F.mse_loss(node_max_prob, graph_prob)

                        loss_distill = mse_loss(st_graph_embed, te_graph_embed)

                        loss = loss_graph + loss_node + args.alpha * loss_distill + args.beta * consistency
                    else:
                        if isinstance(data, HeteroData):
                            x = data['node'].x
                            edge_index = data['node', 'edge', 'node'].edge_index
                            batch = data['node'].batch
                        else:
                            x = data.x
                            edge_index = data.edge_index
                            batch = data.batch

                        graph_out, _ = model(x, labs, edge_index, batch)

                        loss = ce_loss(graph_out, data.y.view(-1))
                    val_loss += loss.item()
                    val_corrects += graph_out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_corrects / len(valid_dataset)

            test_loss = 0.
            test_corrects = 0
            model.eval()
            y_preds = []
            y_tures = []
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    data = data.to(device)

                    if isinstance(data, HeteroData):
                        batch_vector = data['node'].batch
                        inds = torch.bincount(batch_vector).to(device)
                    else:
                        inds = torch.tensor([data.ptr[i + 1] - data.ptr[i] for i in range(data.y.shape[0])]).to(device)

                    labs = torch.repeat_interleave(data.y, inds)

                    if args.train_mode == 'S':
                        if isinstance(data, HeteroData):
                            x = data['node'].x
                            edge_index = data['node', 'edge', 'node'].edge_index
                            batch = data['node'].batch
                        else:
                            x = data.x
                            edge_index = data.edge_index
                            batch = data.batch
                        graph_out, node_out, st_graph_embed = model(x, edge_index, batch)
                        _, te_graph_embed = teacher_model(x, labs, edge_index, batch)
                        loss_distill = mse_loss(st_graph_embed, te_graph_embed)
                        loss_classification = ce_loss(graph_out, data.y.view(-1))
                        loss = loss_classification + args.alpha * loss_distill
                        graph_labels = data.y.view(-1)
                        if args.backbone == 'HGT':
                            node_labels = data['node'].y.view(-1)
                        else:
                            node_labels = data.y_node.view(-1)

                        loss_graph = ce_loss(graph_out, graph_labels)

                        node_mask = node_labels != -1
                        if node_mask.sum() > 0:
                            loss_node = ce_node_loss(node_out[node_mask], node_labels[node_mask])
                        else:
                            soft_label = torch.full_like(node_labels, 0.5).float()
                            soft_label[graph_labels == 0] = 0.0
                            node_probs = torch.softmax(node_out, dim=1)[:, 1]
                            loss_node = F.mse_loss(node_probs, soft_label.float())

                        graph_prob = torch.softmax(graph_out, dim=1)[:, 1]
                        node_prob = torch.softmax(node_out, dim=1)[:, 1]
                        node_max_prob, _ = scatter_max(node_prob, batch, dim=0)
                        consistency = F.mse_loss(node_max_prob, graph_prob)

                        loss_distill = mse_loss(st_graph_embed, te_graph_embed)

                        loss = loss_graph + loss_node + args.alpha * loss_distill + args.beta * consistency
                    else:
                        if isinstance(data, HeteroData):
                            x = data['node'].x
                            edge_index = data['node', 'edge', 'node'].edge_index
                            batch = data['node'].batch
                        else:
                            x = data.x
                            edge_index = data.edge_index
                            batch = data.batch

                        graph_out, _ = model(x, labs, edge_index, batch)

                        loss = ce_loss(graph_out, data.y.view(-1))
                    y_preds.append(graph_out.cpu().numpy())
                    y_tures.append(data.y.view(-1).cpu().numpy())

                    test_loss += loss.item()
                    test_corrects += graph_out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

            test_loss /= len(test_loader)
            test_acc = test_corrects / len(test_dataset)

            if isinstance(y_tures, list):
                y_tures = np.concatenate(y_tures, axis=0)
            if isinstance(y_preds, list):
                y_preds = np.concatenate(y_preds, axis=0)

            if y_preds.ndim > 1:
                if y_preds.shape[1] == 1:
                    probas = y_preds[:, 0]
                else:
                    probas = softmax(y_preds, axis=1)[:, 1]
                y_preds_label = np.argmax(y_preds, axis=1)
            else:
                probas = y_preds
                y_preds_label = (y_preds >= 0.5).astype(int)
            y_tures = y_tures.astype(int)
            y_preds_label = y_preds_label.astype(int)
            try:
                auc = roc_auc_score(y_tures, probas)
            except:
                auc = 0.0
            f1 = f1_score(y_tures, y_preds_label, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_tures, y_preds_label)
            mcc = matthews_corrcoef(y_tures, y_preds_label)

            log = '[*] Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.2f}, Val Loss: {:.3f}, ' \
                  'Val Acc: {:.2f}, Test Loss: {:.3f}, Test Acc: {:.2f}, AUC:{:.3f}, F1:{:.3f}, BA:{:.3f}, MCC:{:.3f}' \
                .format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, auc, f1, balanced_acc,
                        mcc)
            print(log)

            best_balanced_acc = 0
            best_mcc = 0

            if best_test_acc < test_acc:
                best_test_acc = test_acc
            if best_auc < auc:
                best_auc = auc
            if best_f1 < f1:
                best_f1 = f1
            if best_balanced_acc < balanced_acc:
                best_balanced_acc = balanced_acc
            if best_mcc < mcc:
                best_mcc = mcc

            if args.early_stop_metric == 'val_loss':
                current_metric = val_loss
            elif args.early_stop_metric == 'auc':
                current_metric = auc
            elif args.early_stop_metric == 'mcc':
                current_metric = mcc
            elif args.early_stop_metric == 'f1':
                current_metric = f1
            elif args.early_stop_metric == 'ba':
                current_metric = balanced_acc
            elif args.early_stop_metric == 'acc':
                current_metric = test_acc
            else:
                raise ValueError(f"Unknown early_stop_metric: {args.early_stop_metric}")

            if ((args.early_stop_metric == 'val_loss' and current_metric < best_metric) or
                    (args.early_stop_metric != 'val_loss' and current_metric > best_metric)):
                best_metric = current_metric
                wait = 0
                if args.train_mode == 'T':
                    torch.save(model, f'{checkpoints_path}/{args.project}_teacher.pth')
                else:
                    torch.save(model, f'{checkpoints_path}/{args.project}_student.pth')
            else:
                wait += 1

            if wait == args.early_stop:
                print(f'======== Early stopping based on {args.early_stop_metric}! ========')
                break

        test_acc_all.append(best_test_acc)
        auc_all.append(best_auc)
        f1_all.append(best_f1)
        ba_all.append(best_balanced_acc)
        mcc_all.append(best_mcc)

    top_acc = np.asarray(test_acc_all)
    test_avg = np.mean(top_acc)
    test_std = np.std(top_acc)

    print("test_avg_acc: {:.5f}, test_std_acc: {:.5f}, AUC: {:.5f}, F1: {:.5f}, BA: {:.5f}, MCC: {:.5f}".format(
        test_avg,
        test_std,
        max(auc_all),
        max(f1_all),
        max(ba_all),
        max(mcc_all)
    ))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
