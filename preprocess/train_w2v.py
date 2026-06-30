import networkx as nx
from gensim.models import KeyedVectors
import warnings
import argparse
import glob
from multiprocessing import Pool
from functools import partial
import numpy as np
import json
import os, html
import re

warnings.filterwarnings("ignore")
cnt = 0

def tokenize_code_line(line):
    # Sets for operators
    operators3 = {'<<=', '>>='}
    operators2 = {
        '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
        '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
    operators1 = {
        '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
        '=', ',', '?', ':', ';', '{', '}', '!', '~'
    }

    tmp, w = [], []
    i = 0
    if type(i) == None:
        return []
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 3].strip())
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 2].strip())
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []
    # Filter out irrelevant strings
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    return tmp

def joern_to_devign(dot_pdg, word_vectors, out_path):
    """
    Converts PDG to Devign-formatted JSON, synchronized with new Joern label parsing logic.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        
    name = os.path.basename(dot_pdg).replace('.dot', '')
    out_json = os.path.join(out_path, f"{name}.json")
    
    print("===============\t" + dot_pdg)
    
    # Get label (assuming the first character of the filename is the label, e.g., 0_xxx.dot or 1_xxx.dot)
    try:
        vul = int(name[0])
    except ValueError:
        vul = 0 

    # Dynamically get word vector dimension
    embed_size = word_vectors.vector_size if hasattr(word_vectors, 'vector_size') else 128

    node_index = dict()
    node_feature = dict()
    
    try:
        # Read dot file using nx_agraph
        pdg = nx.nx_agraph.read_dot(dot_pdg)
        
        if pdg is not None:
            # 1. Iterate through nodes to extract features
            for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                try:
                    raw_label = pdg.nodes[node].get('label', '')
                    if not raw_label:
                        node_feature[index] = np.zeros(embed_size).tolist()
                        continue
                    
                    # --- Synchronized new version parsing logic ---
                    label = html.unescape(raw_label.strip())
                    # Remove wrapping symbols
                    if (label.startswith('<') and label.endswith('>')) or \
                       (label.startswith('"') and label.endswith('"')):
                        label = label[1:-1]
                    
                    if '<BR/>' in label:
                        parts = label.split('<BR/>')
                        header = parts[0]
                        actual_code = parts[-1]
                        # Extract operator token
                        op_type = header.split(',')[0].strip()
                        op_token = op_type.replace('.', '_').replace('<', '').replace('>', '')
                        combined_text = f"{op_token} {actual_code}"
                    else:
                        # Remove line number after comma
                        combined_text = re.sub(r',\s*\d+$', '', label)

                    # Calculate node feature vector (mean pooling)
                    feature = np.zeros(embed_size)
                    tokens = tokenize_code_line(combined_text)
                    token_count = 0
                    
                    for token in tokens:
                        if token in word_vectors:
                            feature += word_vectors[token]
                            token_count += 1
                    
                    if token_count > 0:
                        feature = feature / token_count # Average to prevent numerical explosion for long code
                    
                    node_feature[index] = feature.tolist()
                except Exception:
                    node_feature[index] = np.zeros(embed_size).tolist()
                    continue

            # 2. Construct node feature list
            nodes_ = []
            for i in range(len(pdg.nodes())):
                if i in node_feature:
                    nodes_.append(node_feature[i])
                else:
                    nodes_.append(np.zeros(embed_size).tolist())

            # 3. Construct edge list (Devign format: source, type, destination)
            edges_ = []
            for s, d in pdg.edges():
                # Default all edge types to 0
                if s in node_index and d in node_index:
                    edges_.append((node_index[s], 0, node_index[d]))

            # 4. Save data
            if nodes_ and edges_:
                data = {
                    'node_features': nodes_,
                    'graph': edges_,
                    'target': vul
                }
                with open(out_json, 'w') as f:
                    json.dump(data, f)
    except Exception as e:
        print(f"Error processing {dot_pdg}: {e}")
        
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser', default='3_export/pdg_norm/0_novul')
    parser.add_argument('--output_dir', type=str, help='Output Directory of the parser', default='4_embedding/pdg_norm/')
    args = parser.parse_args()

    out_path = args.output_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    dots = glob.glob(args.input_dir + '/*.dot')
    
    # Load Word2Vec model
    word_vectors = KeyedVectors.load('3_export/pdg_norm/primevul.wv', mmap='r')
    
    # Process in parallel
    pool = Pool(4)
    pool.map(partial(joern_to_devign, word_vectors=word_vectors, out_path=out_path), dots)

if __name__ == '__main__':
    main()