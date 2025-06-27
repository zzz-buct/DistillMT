from script.my_util import *
import os
import pandas as pd
import re
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

csv_data_path = '../datasets/preprocessed_method_data_callee'
embedding_save_path = '../datasets/embedding/'
bert_models_path = '../bert_models/'
embedding_length = 50

os.makedirs(embedding_save_path, exist_ok=True)

char_to_remove = ['+', '-', '*', '/', '=', '++', '--', '\\', '<str>', '<char>', '|', '&', '!']


def preprocess_code_line(code_line):
    code_line = re.sub("''", "'", code_line)
    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)
    code_line = re.sub(r'\b\d+\b', '', code_line)
    code_line = re.sub(r"\[.*?\]", '', code_line)
    code_line = re.sub(r"[\.|,|:|;|{|}|\(|\)]", ' ', code_line)
    for char in char_to_remove:
        code_line = code_line.replace(char, ' ')
    return code_line.strip()


def load_and_preprocess_csv(filepath):
    df = pd.read_csv(filepath)
    df['processed_code'] = df['code_line'].astype(str).apply(preprocess_code_line)
    df['tokens'] = df['processed_code'].apply(lambda x: x.split())
    return df


def train_word2vec(sentences, dim=embedding_length):
    model = Word2Vec(sentences=sentences, vector_size=dim, window=5, min_count=1, workers=4)
    return model


def embed_code_lines(df, model, dim=embedding_length):
    def embed(tokens):
        vectors = []
        for tok in tokens:
            if tok in model.wv:
                vectors.append(model.wv[tok])
            else:
                vectors.append(model.wv["<unk>"])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(dim)

    return df['tokens'].apply(embed)


def process_project(all_releases, method='Word2Vec', dim=50):
    for proj, cur_all_rel in all_releases.items():
        for rel in cur_all_rel:
            csv_file = os.path.join(csv_data_path, f"{rel}_with_callee.csv")
            if not os.path.exists(csv_file):
                print(f"File not found: {csv_file}")
                continue
            print(f"Processing: {csv_file} using method {method}")
            df = pd.read_csv(csv_file)

            if method in ['Word2Vec', 'Doc2Vec']:
                df['processed_code'] = df['code_line'].astype(str).apply(preprocess_code_line)
                df['tokens'] = df['processed_code'].apply(lambda x: x.split())
            else:
                df['processed_code'] = df['code_line'].astype(str)

            if method == 'Word2Vec':
                if rel == cur_all_rel[0]:
                    print("Training Word2Vec model...")
                    sentences = df['tokens'].tolist()
                    sentences.append(["<unk>"])
                    w2v_model = train_word2vec(sentences, dim)

                    def get_token_vector(tok):
                        return w2v_model.wv[tok] if tok in w2v_model.wv else w2v_model.wv["<unk>"]
                embed_fn = lambda tokens: np.mean([get_token_vector(tok) for tok in tokens],
                                                  axis=0) if tokens else np.zeros(dim)
                df['embedding'] = df['tokens'].apply(embed_fn)

            elif method == 'Doc2Vec':
                if rel == cur_all_rel[0]:
                    print("Training Doc2Vec model...")
                    documents = [TaggedDocument(words=toks, tags=[i]) for i, toks in enumerate(df['tokens'].tolist())]
                    d2v_model = Doc2Vec(documents, vector_size=dim, window=5, min_count=1, workers=4, epochs=20)
                df['embedding'] = [d2v_model.infer_vector(tokens) for tokens in df['tokens']]


            elif method in ['CodeBERT', 'GraphCodeBERT']:
                model_path = os.path.join(bert_models_path, "CodeBert" if method == 'CodeBERT' else "GraphCodeBert")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                model.eval()

                def codebert_embed(text):
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
                    return cls_embedding[:dim] if cls_embedding.shape[0] > dim else np.pad(cls_embedding, (

                        0, dim - cls_embedding.shape[0]), 'constant')

                tqdm.pandas(desc=f"Encoding with {method}")

                df['embedding'] = df['processed_code'].progress_apply(codebert_embed)

            embed_df = df[['filename', 'code_line', 'line_number']].copy()
            embed_df = pd.concat([embed_df, pd.DataFrame(df['embedding'].tolist(),
                                                         columns=[f'x_{i + 1}' for i in range(dim)])], axis=1)

            method_dir = os.path.join(embedding_save_path, method)
            os.makedirs(method_dir, exist_ok=True)
            save_path = os.path.join(method_dir, f"{rel}.csv")
            embed_df.to_csv(save_path, index=False)
            print(f"Saved embeddings to: {save_path}")


if __name__ == '__main__':
    # process_project(all_releases=all_releases, method='Word2Vec', dim=embedding_length)
    # process_project(all_releases=all_releases, method='Doc2Vec', dim=embedding_length)
    # process_project(all_releases=all_releases, method='CodeBERT', dim=embedding_length)
    process_project(all_releases=all_releases, method='GraphCodeBERT', dim=embedding_length)
