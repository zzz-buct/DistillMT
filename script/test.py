import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef, recall_score, \
    precision_score
from torch_geometric.data import DataLoader, HeteroData
from MyDataset import MyDataset
from MyHeteroDataset import MyHeteroDataset
import warnings

warnings.filterwarnings("ignore")

within_path = './results/within-project/'
cross_path = './results/cross-project/'


def evaluate(model, loader, device, is_hetero):
    model.eval()
    graph_y_preds, graph_y_trues = [], []
    node_y_preds, node_y_trues = [], []

    graph_loss_total, node_loss_total = 0.0, 0.0
    graph_loss_fn = torch.nn.CrossEntropyLoss()
    node_loss_fn = torch.nn.CrossEntropyLoss()

    graph_records = []
    node_records = []

    with torch.no_grad():
        graph_id_counter = 0
        node_id_counter = 0

        for data in loader:
            data = data.to(device)

            if is_hetero:
                x = data['node'].x
                edge_index = data['node', 'edge', 'node'].edge_index
                batch = data['node'].batch
                node_labels = data['node'].y.view(-1)
            else:
                x = data.x
                edge_index = data.edge_index
                batch = data.batch
                node_labels = data.y_node.view(-1)

            graph_out, node_out, _ = model(x, edge_index, batch)

            graph_labels = data.y.view(-1)
            graph_loss = graph_loss_fn(graph_out, graph_labels)
            graph_loss_total += graph_loss.item()

            graph_pred_np = graph_out.cpu().numpy()
            graph_label_np = graph_labels.cpu().numpy()
            graph_y_preds.append(graph_pred_np)
            graph_y_trues.append(graph_label_np)

            probas = softmax(graph_pred_np, axis=1)[:, 1]
            preds = np.argmax(graph_pred_np, axis=1)
            for i in range(len(graph_labels)):
                graph_records.append({
                    "graph_id": graph_id_counter,
                    "true_label": int(graph_label_np[i]),
                    "pred_label": int(preds[i]),
                    "prob": float(probas[i])
                })
                graph_id_counter += 1

            if node_labels is not None:
                node_mask = node_labels != -1
                if node_mask.sum() > 0:
                    node_loss = node_loss_fn(node_out[node_mask], node_labels[node_mask])
                    node_loss_total += node_loss.item()

                    node_probs = torch.softmax(node_out[node_mask], dim=1)
                    node_y_preds.append(node_probs[:, 1].cpu().numpy())
                    node_y_trues.append(node_labels[node_mask].cpu().numpy())

                    # 获取对应节点的 batch 图索引
                    graph_true_labels = graph_labels.cpu().numpy()
                    graph_pred_labels = preds
                    graph_probs = probas

                    graph_id_offset = graph_id_counter
                    for i, prob in enumerate(node_probs[:, 1]):
                        g_id_in_batch = batch[node_mask][i].item()
                        g_id = graph_id_offset + g_id_in_batch

                        node_records.append({
                            "node_index": node_id_counter,
                            "node_true_label": int(node_labels[node_mask][i].item()),
                            "node_pred_label": int((prob >= 0.5)),
                            "node_prob": float(prob.item()),
                            "graph_index": int(g_id),
                            "graph_true_label": int(graph_true_labels[g_id_in_batch]),
                            "graph_pred_label": int(graph_pred_labels[g_id_in_batch]),
                            "graph_prob": float(graph_probs[g_id_in_batch])
                        })
                        node_id_counter += 1
                    graph_id_counter += batch.max().item() + 1

    graph_y_preds = np.concatenate(graph_y_preds, axis=0)
    graph_y_trues = np.concatenate(graph_y_trues, axis=0)

    graph_probas = softmax(graph_y_preds, axis=1)[:, 1]
    graph_y_preds_label = np.argmax(graph_y_preds, axis=1)

    graph_y_preds_label = graph_y_preds_label.astype(int)
    graph_y_trues = graph_y_trues.astype(int)

    try:
        graph_auc = roc_auc_score(graph_y_trues, graph_probas)
    except:
        graph_auc = 0.0

    graph_precision = precision_score(graph_y_trues, graph_y_preds_label)
    graph_recall = recall_score(graph_y_trues, graph_y_preds_label)
    graph_f1 = f1_score(graph_y_trues, graph_y_preds_label, zero_division=0)
    graph_ba = balanced_accuracy_score(graph_y_trues, graph_y_preds_label)
    graph_mcc = matthews_corrcoef(graph_y_trues, graph_y_preds_label)
    graph_acc = (graph_y_preds_label == graph_y_trues).sum() / len(graph_y_trues)

    if len(node_y_trues) > 0:
        node_y_trues = np.concatenate(node_y_trues, axis=0)
        node_y_preds = np.concatenate(node_y_preds, axis=0)
        node_y_preds_label = (node_y_preds >= 0.5).astype(int)

        try:
            node_auc = roc_auc_score(node_y_trues, node_y_preds)
        except:
            node_auc = 0.0

        node_f1 = f1_score(node_y_trues, node_y_preds_label, zero_division=0)
        node_ba = balanced_accuracy_score(node_y_trues, node_y_preds_label)
        node_mcc = matthews_corrcoef(node_y_trues, node_y_preds_label)
        node_acc = (node_y_preds_label == node_y_trues).sum() / len(node_y_trues)
        node_loss_avg = node_loss_total / len(loader)
    else:
        node_acc = node_auc = node_f1 = node_ba = node_mcc = node_loss_avg = -1

    return (
        graph_precision, graph_recall, graph_acc, graph_auc, graph_f1, graph_ba, graph_mcc,
        graph_loss_total / len(loader),
        node_acc, node_auc, node_f1, node_ba, node_mcc, node_loss_avg,
        graph_records, node_records)


def main():
    parser = argparse.ArgumentParser(description='Test pre-trained model on test dataset')
    parser.add_argument('--model_project', type=str, required=True, help='project name (i.e., dataset folder)')
    parser.add_argument('--target_project', type=str, required=True, help='project name (i.e., dataset folder)')
    parser.add_argument('--test_dataset', type=str, required=True, help='name of test dataset file')
    parser.add_argument('--backbone', type=str, default='GCN', help='GNN backbone: GCN, GAT, HGT, etc.')
    parser.add_argument('--device', type=int, default=0, help='cuda device number')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for testing')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='path to model checkpoints')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    is_hetero = 'HGT' in args.backbone

    if is_hetero:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './HeteroData', args.target_project)
        test_dataset = MyHeteroDataset(path, name=args.test_dataset)
    else:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './data', args.target_project)
        test_dataset = MyDataset(path, name=args.test_dataset, use_node_attr=True, use_edge_attr=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    x = test_dataset[0]['node'].x if is_hetero else test_dataset[0].x
    input_dim = x.size(1)
    num_classes = len(set([data.y.item() for data in test_dataset]))

    if args.model_project == args.target_project:
        print(f"==== Within-Project Prediction: Source Model:{args.model_project}, "
              f"Test Results for {args.test_dataset} ====")
    else:
        print(f"==== Cross-Project Prediction: Source Model:{args.model_project} -> {args.target_project}, "
              f"Test Results for {args.test_dataset} ====")

    model_path = os.path.join(args.checkpoints_path, args.backbone, f"{args.model_project}_student.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    (graph_precision, graph_recall, graph_acc, graph_auc, graph_f1, graph_ba, graph_mcc, graph_loss,
     node_acc, node_auc, node_f1, node_ba, node_mcc, node_loss,
     graph_records, node_records) = evaluate(model, test_loader, device, is_hetero)
    if len(node_records) > 0:
        df_line = pd.DataFrame(node_records)
        df_sorted = df_line.sort_values(by="node_prob", ascending=False).reset_index(drop=True)

        # Recall@Top20%LOC
        top_20_percent = int(0.2 * len(df_sorted))
        recall_20_loc = recall_score(
            df_sorted["node_true_label"],
            [1 if i < top_20_percent else 0 for i in range(len(df_sorted))]
        )

        # Effort@Top20%Recall
        total_defect = df_sorted["node_true_label"].sum()
        target_tp = int(total_defect * 0.2)
        tp_found = 0
        effort_line_index = 0
        for i, row in df_sorted.iterrows():
            if row["node_true_label"]:
                tp_found += 1
                if tp_found >= target_tp:
                    effort_line_index = i
                    break
        effort_20_recall = effort_line_index / len(df_sorted) if len(df_sorted) > 0 else 0

        # Initial False Alarms (IFA)
        ifa_list = []
        defective_files = df_sorted[(df_sorted["graph_true_label"] == True) & (df_sorted['graph_pred_label'] == 1)][
            "graph_index"].unique()
        for file in defective_files:
            file_df = df_sorted[df_sorted["graph_index"] == file].sort_values(by="node_prob", ascending=False)
            for i, (_, row) in enumerate(file_df.iterrows()):
                if row["node_true_label"]:
                    ifa_list.append(i)
                    break
        ifa_score = np.mean(ifa_list) if ifa_list else 0

    print(
        f"[File-Level] Acc: {graph_acc:.4f}, AUC: {graph_auc:.4f}, BA: {graph_ba:.4f}, MCC: {graph_mcc:.4f}, Loss: {graph_loss:.4f}")
    print(
        f"[Line-Level] Recall@Top20%LOC: {recall_20_loc:.4f}, Effort@Top20%Recall: {effort_20_recall:.4f}, IFA: {ifa_score:.4f}")
    # print('\n')

    if args.model_project == args.target_project:
        result_path = within_path + f'{args.backbone}'
        if not os.path.exists(f"{result_path}/{args.target_project}"):
            os.makedirs(f"{result_path}/{args.target_project}", exist_ok=True)

        pd.DataFrame(graph_records).to_csv(
            f"{result_path}/{args.target_project}/{args.test_dataset}_graph_predictions.csv",
            index=False)
        if len(node_records) > 0:
            pd.DataFrame(node_records).to_csv(
                f"{result_path}/{args.model_project}/{args.test_dataset}_node_predictions.csv", index=False)
    else:
        result_path = cross_path + f'{args.backbone}'
        if not os.path.exists(f"{result_path}/{args.target_project}"):
            os.makedirs(f"{result_path}/{args.target_project}", exist_ok=True)

        pd.DataFrame(graph_records).to_csv(
            f"{result_path}/{args.target_project}/{args.model_project}-{args.test_dataset}_graph_predictions.csv",
            index=False)
        if len(node_records) > 0:
            pd.DataFrame(node_records).to_csv(
                f"{result_path}/{args.target_project}/{args.model_project}-{args.test_dataset}_node_predictions.csv",
                index=False)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time: {end - start}')
