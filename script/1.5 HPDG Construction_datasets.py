import os, re
from script.my_util import *
import warnings

warnings.filterwarnings("ignore")

csv_dir = '../datasets/preprocessed_data_with_node_type'
embedding_dir = '../datasets/embedding'
sourcecode_dir = "../sourcecode/"

save_csv_dir = '../datasets/TUDataset/'


def get_TUDataset_format(proj, embedding_method='GraphCodeBert'):
    os.makedirs(save_csv_dir, exist_ok=True)
    node_type_map = {}
    node_type_counter = 0

    cur_all_rel = all_releases[proj]
    for rel in cur_all_rel:
        print(f"Processing release: {rel}")

        df_embedding = pd.read_csv(f'{embedding_dir}/{embedding_method}/{rel}.csv')
        df_rel = pd.read_csv(f'{csv_dir}/{rel}.csv')

        global_node_id = 1
        global_graph_id = 1

        A_lines = []
        graph_indicator = []
        graph_labels = []
        node_labels = []
        edge_labels = []
        node_attributes = []

        file_order = df_rel['filename'].drop_duplicates().tolist()
        for filename in file_order:
            group = df_rel[df_rel['filename'] == filename]
            java_path = f'{sourcecode_dir}/{rel}/{filename}'
            edges_path = java_path.replace('.java', '_edges.txt')
            edge_labels_path = java_path.replace('.java', '_edge_labels.txt')

            if not os.path.exists(edges_path) or not os.path.exists(edge_labels_path):
                print(f"Skipping missing file: {filename}")
                continue

            local_node_count = len(group)
            index_map = {}

            for i in range(local_node_count):
                graph_indicator.append(global_graph_id)
                index_map[group.iloc[i]['line_number']] = global_node_id

                label = group.iloc[i]['line-label']
                node_labels.append(1 if label else 0)

                embedding_cols = [col for col in df_embedding.columns if col.startswith('x_')]
                emb_row = df_embedding[
                    (df_embedding['filename'] == filename) &
                    (df_embedding['line_number'] == group.iloc[i]['line_number'])
                    ]

                if not emb_row.empty:
                    emb_vec = emb_row.iloc[0][embedding_cols].values.astype(float).tolist()
                else:
                    emb_vec = [0.0] * len(embedding_cols)

                node_type = group.iloc[i]['node_type']
                if node_type not in node_type_map:
                    node_type_map[node_type] = node_type_counter
                    node_type_counter += 1
                node_type_id = node_type_map[node_type]

                emb_vec.append(node_type_id)
                node_attributes.append(emb_vec)

                global_node_id += 1

            graph_labels.append(1 if group.iloc[0]['file-label'] else 0)

            edge_df = pd.read_csv(edges_path, sep=' ', header=None)
            edge_label_df = pd.read_csv(edge_labels_path, header=None)

            for idx, row in edge_df.iterrows():
                src_line, tgt_line = row[0], row[1]
                if src_line in index_map and tgt_line in index_map:
                    A_lines.append(f"{index_map[src_line]},{index_map[tgt_line]}")
                    edge_labels.append(edge_label_df.iloc[idx][0])

            global_graph_id += 1

        rel_save_dir = os.path.join(save_csv_dir, proj, rel, 'raw')
        os.makedirs(rel_save_dir, exist_ok=True)

        write_list_to_file(os.path.join(rel_save_dir, f"{rel}_A.txt"), A_lines)
        write_list_to_file(os.path.join(rel_save_dir, f"{rel}_graph_indicator.txt"), graph_indicator)
        write_list_to_file(os.path.join(rel_save_dir, f"{rel}_graph_labels.txt"), graph_labels)
        write_list_to_file(os.path.join(rel_save_dir, f"{rel}_node_labels.txt"), node_labels)
        write_list_to_file(os.path.join(rel_save_dir, f"{rel}_edge_labels.txt"), edge_labels)
        write_list_of_lists_to_file(os.path.join(rel_save_dir, f"{rel}_node_attributes.txt"), node_attributes)

        print(f"{rel} saved to {rel_save_dir}")


def write_list_to_file(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(str(item) + "\n")


def write_list_of_lists_to_file(path, data):
    with open(path, "w") as f:
        for row in data:
            line = ",".join(map(str, row))
            f.write(line + "\n")


if __name__ == '__main__':
    for proj in list(all_releases.keys()):
        get_TUDataset_format(proj=proj, embedding_method='GraphCodeBert')
