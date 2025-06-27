import os, re
from script.my_util import *
import warnings

warnings.filterwarnings("ignore")

csv_dir = '../datasets/preprocessed_data_no_comment/'
sourcecode_dir = "../sourcecode/"
save_csv_dir = '../datasets/preprocessed_data_with_node_type/'

if not os.path.exists(save_csv_dir):
    os.makedirs(save_csv_dir)

java_types = ['int', 'long', 'double', 'float', 'boolean', 'char', 'byte', 'short',
              'String', 'List', 'Map', 'Set', 'HashMap', 'ArrayList', 'LinkedList',
              'Connection', 'URI', 'Object']


def determine_node_types(df: pd.DataFrame):
    node_types = [''] * len(df)
    block_stack = []
    last_stmt_idx_stack = []

    for i, row in df.iterrows():
        code = str(row['code_line']).strip()
        first_token = code.split()[0] if code else ''

        if row['is_blank'] or row['is_comment']:
            node_types[i] = 'FALLBACK_NODE'
            continue
        if ('main' in code and 'String[]' in code) or re.match(r'^\s*(public|private|protected)?\s*class\s+\w+', code):
            node_types[i] = 'ENTRY_POINT'
        elif re.match(r'^\s*return\b', code):
            node_types[i] = 'RETURN'
        elif re.match(r'^\s*package\s+[a-zA-Z0-9_.]+;', code) or re.match(r'^\s*import\s+[a-zA-Z0-9_.\*]+;', code):
            node_types[i] = 'FALLBACK_NODE'
        elif re.match(r'(public|private|protected)?\s*(static\s+)?[\w<>]+\s+\w+\s*\(.*\)\s*{?$', code) or (
                re.search(r'\b\w+\s*\(.*\)', code)
        ):
            node_types[i] = 'FUNCTION_NAME'
        elif re.match(r'^if\s*\(.*\)\s*{?$', code):
            node_types[i] = 'IF'
            block_stack.append('if')
            last_stmt_idx_stack.append(None)
        elif re.match(r'^(for|while)\s*\(.*\)\s*{?$', code):
            node_types[i] = 'LOOP'
            block_stack.append('loop')
            last_stmt_idx_stack.append(None)
        elif ';' in code:
            node_types[i] = 'EXPRESSION'
            if block_stack and last_stmt_idx_stack[-1] is None:
                last_stmt_idx_stack[-1] = i
        elif first_token in java_types:
            node_types[i] = 'NEW_VARIABLE'
            if block_stack and last_stmt_idx_stack[-1] is None:
                last_stmt_idx_stack[-1] = i
        else:
            node_types[i] = 'FALLBACK_NODE'

        if block_stack and node_types[i] not in {'IF', 'LOOP'}:
            last_stmt_idx = last_stmt_idx_stack[-1]
            if last_stmt_idx is not None:
                block_type = block_stack.pop()
                last_stmt_idx_stack.pop()
                if block_type == 'if':
                    node_types[last_stmt_idx] = 'END_IF'
                elif block_type == 'loop':
                    node_types[last_stmt_idx] = 'END_LOOP'

    df['node_type'] = node_types
    return df


def get_all_project_nodes(proj):
    cur_all_rel = all_releases[proj]
    for rel in cur_all_rel:
        df_rel = pd.read_csv(f'{csv_dir}/{rel}_with_callee.csv')
        df_with_types = determine_node_types(df_rel)
        df_with_types.to_csv(f'{save_csv_dir}/{rel}.csv', index=False)
        print(f'{rel} done')


if __name__ == '__main__':
    for proj in list(all_releases.keys()):
        get_all_project_nodes(proj)
