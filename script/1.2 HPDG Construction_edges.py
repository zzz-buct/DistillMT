from script.my_util import *
import pygraphviz as pgv
import warnings

warnings.filterwarnings("ignore")

csv_dir = '../datasets/preprocessed_data_no_comment/'
sourcecode_dir = "../sourcecode/"


def extract_numbers_from_label(code_line):
    numbers = []
    single_number_pattern = r'<(\d+)>'
    match = re.search(single_number_pattern, code_line)
    if match:
        numbers.append(int(match.group(1)))
    range_pattern = r'<(\d+)\s*\.\.\.\s*(\d+)>'
    match = re.search(range_pattern, code_line)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        numbers.extend(range(start, end + 1))
    return numbers


def get_edges_list(dot_file_path, java_file_path):
    AGraph = pgv.AGraph(dot_file_path)
    nodes_to_lineNumbers = {}
    nodes = AGraph.nodes()
    for node in nodes:
        label = str(AGraph.get_node(node).attr['label'])
        if 'Enter <' in label:
            range_match = re.search(r'Enter <(\d+)\.\.\.(-?\d+)>', label)
            single_match = re.search(r'Enter <(\d+)>', label)
            if range_match:
                start_line = int(range_match.group(1))
                nodes_to_lineNumbers[node] = [start_line]
            elif single_match:
                start_line = int(single_match.group(1))
                nodes_to_lineNumbers[node] = [start_line]
        else:
            nodes_to_lineNumbers[node] = extract_numbers_from_label(AGraph.get_node(node).attr['label'])

    with open(java_file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    method_ranges = []
    for subgraph in AGraph.subgraphs():
        subgraph_label = str(subgraph.graph_attr.get('label', 'None'))
        line_numbers = re.findall(r'<(\d+)\.\.\.(\d+)>', subgraph_label)
        if line_numbers:
            start_line, end_line = map(int, line_numbers[0])
            method_ranges.append((start_line, end_line))

    edge_labels = []
    mapped_edges = []
    original_edges = AGraph.edges()
    connected_lines = set()

    for edge in original_edges:
        edge_label = '2'
        if 'solid' in edge.attr['style']:
            edge_label = '0'
        elif 'bold' in edge.attr['style']:
            edge_label = '3'
        elif 'dotted' in edge.attr['style']:
            if 'true' in edge.attr['label']:
                edge_label = '1'
            elif 'false' in edge.attr['label']:
                edge_label = '2'

        src_node, dst_node = edge
        src_lines = nodes_to_lineNumbers.get(src_node, [])
        dst_lines = nodes_to_lineNumbers.get(dst_node, [])

        for src_line in src_lines:
            for dst_line in dst_lines:
                src_line = int(src_line)
                dst_line = int(dst_line)
                mapped_edges.append([src_line, dst_line])
                edge_labels.append(edge_label)

                connected_lines.add(src_line)
                connected_lines.add(dst_line)

    method_lines = set()
    method_ranges = sorted(method_ranges, key=lambda x: x[0])
    prev_end = 0
    for start, end in method_ranges:
        non_method_start = prev_end + 1
        non_method_end = start - 1
        if non_method_start <= non_method_end:
            for line in range(non_method_start, non_method_end + 1):
                if line < non_method_end:
                    edge = [line, line + 1]
                    if edge not in mapped_edges:
                        mapped_edges.append(edge)
                        edge_labels.append('4')
        prev_end = end
    if prev_end < total_lines:
        for line in range(prev_end + 1, total_lines):
            edge = [line, line + 1]
            if edge not in mapped_edges:
                mapped_edges.append(edge)
                edge_labels.append('4')

    non_method_lines = [line for line in range(1, total_lines + 1) if line not in method_lines]
    non_method_lines.sort()
    for i in range(len(non_method_lines) - 1):
        src_line = non_method_lines[i]
        dst_line = non_method_lines[i + 1]

        if dst_line == src_line + 1:
            edge = [src_line, dst_line]
            if edge not in mapped_edges:
                mapped_edges.append(edge)
                edge_labels.append('4')

    unique_edge_set = set()
    unique_edges = []
    unique_labels = []

    for edge, label in zip(mapped_edges, edge_labels):
        edge_key = (edge[0], edge[1], label)
        if edge_key not in unique_edge_set:
            unique_edge_set.add(edge_key)
            unique_edges.append(edge)
            unique_labels.append(label)

    sorted_edges_with_labels = sorted(zip(unique_edges, unique_labels), key=lambda x: x[0][0])
    unique_edges, unique_labels = zip(*sorted_edges_with_labels)

    return unique_edges, unique_labels


def get_all_project_edges(proj_name):
    cur_all_rel = all_releases[proj_name]
    for rel in cur_all_rel:
        df_rel = pd.read_csv(f'{csv_dir}/{rel}_with_callee.csv')
        grouped = df_rel.groupby('filename')
        for filename, group in grouped:
            java_path = f'{sourcecode_dir}/{rel}/{filename}'
            dot_file_path = java_path.replace('.java', '_pdg.dot')

            unique_edges, unique_labels = get_edges_list(dot_file_path, java_path)
            if len(unique_edges) == len(unique_labels):
                edges_path = java_path.replace('.java', '_edges.txt')
                with open(edges_path, 'w', encoding='utf-8') as f:
                    for edge in unique_edges:
                        f.write(f"{edge[0]} {edge[1]}\n")
                edge_labels_path = java_path.replace('.java', '_edge_labels.txt')
                with open(edge_labels_path, 'w', encoding='utf-8') as f:
                    for label in unique_labels:
                        f.write(f"{label}\n")
            else:
                print('error:', filename)
        print(f'{rel} done')


if __name__ == '__main__':
    for proj in list(all_releases.keys()):
        get_all_project_edges(proj)
