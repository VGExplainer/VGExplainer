import networkx as nx
from gensim.models import KeyedVectors
import warnings
import argparse
import glob
from multiprocessing import Pool
from functools import partial
import numpy as np
import json
import os,html
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
    将 PDG 转换为 Devign 格式的 JSON，同步新版本 Joern 标签解析逻辑
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
        
    name = os.path.basename(dot_pdg).replace('.dot', '')
    out_json = os.path.join(out_path, f"{name}.json")
    
    print("===============\t" + dot_pdg)
    
    # 获取标签 (假设文件名第一个字符是标签，如 0_xxx.dot 或 1_xxx.dot)
    try:
        vul = int(name[0])
    except ValueError:
        vul = 0 

    # 动态获取词向量维度 (之前建议设为 128)
    embed_size = word_vectors.vector_size if hasattr(word_vectors, 'vector_size') else 128

    node_index = dict()
    node_feature = dict()
    
    try:
        # 使用 nx_agraph 读取 dot
        pdg = nx.nx_agraph.read_dot(dot_pdg)
        
        if pdg is not None:
            # 1. 遍历节点提取特征
            for index, node in enumerate(pdg.nodes()):
                node_index[node] = index
                try:
                    raw_label = pdg.nodes[node].get('label', '')
                    if not raw_label:
                        node_feature[index] = np.zeros(embed_size).tolist()
                        continue
                    
                    # --- 同步新版本解析逻辑 ---
                    label = html.unescape(raw_label.strip())
                    # 去掉包裹符号
                    if (label.startswith('<') and label.endswith('>')) or \
                       (label.startswith('"') and label.endswith('"')):
                        label = label[1:-1]
                    
                    if '<BR/>' in label:
                        parts = label.split('<BR/>')
                        header = parts[0]
                        actual_code = parts[-1]
                        # 提取操作符 Token
                        op_type = header.split(',')[0].strip()
                        op_token = op_type.replace('.', '_').replace('<', '').replace('>', '')
                        combined_text = f"{op_token} {actual_code}"
                    else:
                        # 移除逗号后的行号
                        combined_text = re.sub(r',\s*\d+$', '', label)

                    # 计算节点特征向量 (均值处理)
                    feature = np.zeros(embed_size)
                    tokens = tokenize_code_line(combined_text)
                    token_count = 0
                    
                    for token in tokens:
                        if token in word_vectors:
                            feature += word_vectors[token]
                            token_count += 1
                    
                    if token_count > 0:
                        feature = feature / token_count # 平均值化，防止长代码数值爆炸
                    
                    node_feature[index] = feature.tolist()
                except Exception:
                    node_feature[index] = np.zeros(embed_size).tolist()
                    continue

            # 2. 构造节点特征列表
            nodes_ = []
            for i in range(len(pdg.nodes())):
                if i in node_feature:
                    nodes_.append(node_feature[i])
                else:
                    nodes_.append(np.zeros(embed_size).tolist())

            # 3. 构造边列表 (Devign 格式: s, type, d)
            edges_ = []
            for s, d in pdg.edges():
                # 默认所有边类型为 0
                if s in node_index and d in node_index:
                    edges_.append((node_index[s], 0, node_index[d]))

            # 4. 保存数据
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

# def joern_to_devign(dot_pdg, word_vectors, out_path):
#     #out_path = out_path + '1_CVE'+dot_pdg.split("/")[6].split('@')[0].split('CVE')[-1]+'/'
#     if not os.path.exists(out_path):
#        os.mkdir(out_path)
#     name = dot_pdg.split('/')[-1].split('.dot')[0]
#     out_json = out_path + name + '.json'
#     # if os.path.exists(out_json):
#     #     print("-----> has been processed :\t", out_json)
#     #     return
#     print("===============\t"+dot_pdg)
#     vul = int(name[0])
#     node_index = dict()
#     node_feature = dict()
#     try:
#         # pdg = nx.drawing.nx_pydot.read_dot(dot_pdg)
#         pdg = nx.nx_agraph.read_dot(dot_pdg)
#         if type(pdg) != None:
#             for index, node in enumerate(pdg.nodes()):
#                 node_index[node] = index
#                 try:
#                     label = pdg.nodes[node]['label'][1:-1]
#                     code = label.partition(',')[2]
#                     feature = np.array([0.0 for i in range(100)])
#                     for token in tokenize_code_line(code):
#                         if token in word_vectors:
#                             feature += np.array(word_vectors[token])
#                         else:
#                             feature += np.array([0.0 for i in range(100)])
#                     node_feature[index] = feature
#                 except:
#                     continue

#             nodes_ = []
#             for i in range(len(list(pdg.nodes()))):
#                 try:
#                     nodes_.append(list(node_feature[i]))
#                 except:
#                     continue

#             edges_ = []
#             for item in pdg.adj.items():
#                 s = item[0]
#                 for edge_relation in item[1]:
#                     d = edge_relation    
#                     ddg_flag = 0
#                     cdg_flag = 0 
#                     edges_.append((node_index[s],0, node_index[d]))
#                     # for edge in item[1]._atlas[edge_relation].items():
#                     #     if 'CFG:' in edge[1]['label'] and ddg_flag == 0:
#                     #         edge_type = 0
#                     #         ddg_flag = 1
#                     #         edges_.append((node_index[s], edge_type, node_index[d]))
#                     #     if 'DDG:' in edge[1]['label'] and ddg_flag == 0:
#                     #         edge_type = 0
#                     #         ddg_flag = 1
#                     #         edges_.append((node_index[s], edge_type, node_index[d]))
#                     #     elif 'CDG:' in edge[1]['label'] and cdg_flag == 0:
#                     #         edge_type = 1
#                     #         cdg_flag = 1
#                     #         edges_.append((node_index[s], edge_type, node_index[d]))

#                     #     edge_type = 0
#                     #     edges_.append((node_index[s], edge_type, node_index[d]))

#             data = dict()
#             if nodes_== [] or edges_ ==[]:
#                 return
#             data['node_features'] = nodes_
#             data['graph'] = edges_
#             data['target'] = vul
#             out_json = out_path + name + '.json'
#             with open(out_json, 'w') as f:
#                 f.write(json.dumps(data))  

#     except:
#          pass
#     return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',default='/home/Dataset/primevul/3_export/pdg_norm/0_novul_j4')
    parser.add_argument('--output_dir', type=str, help='Output Directory of the parser',default='/home/Dataset/primevul/4_embedding/j4/pdg_norm/')
    args = parser.parse_args()

    dir_path_list = [args.input_dir]
    out_path = args.output_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dots = []
    dots = glob.glob(args.input_dir+ '/*.dot')
    # dots += glob.glob("/home/Dataset/devign/5_cpg/1_vul"+ '/*.dot')
    #dots = dots + glob.glob('/home/Dataset/msr/new_slice/vul_complete'+ '/*.dot')
    # for dir_path in dir_path_list:
    #     dots_tmp = glob.glob(dir_path + '*.dot')
    #     for dot in dots_tmp:
    #         if '-' in dot.rsplit('/')[-1]: #说明是从joern直接解析的结果，没有重新命名过
    #             if 'novul' in dir_path: #无漏洞的为0_开头
    #                 new_name = dot[:dot.rindex('/')+1] + '0_' + dot[dot.rindex('/')+1:].replace("-pdg","")
    #             else: # 有漏洞的为1_开头
    #                 new_name = dot[:dot.rindex('/')+1] + '1_' + dot[dot.rindex('/')+1:].replace("-pdg","")
    #             os.system("mv "+ dot + ' ' + new_name)
    #         else:
    #             new_name = dot
    #         dots.append(new_name)
    #读取词向量模型w2v
    word_vectors = KeyedVectors.load('/home/Dataset/primevul/3_export/pdg_norm/primevul.wv', mmap='r')
    #word_vectors = KeyedVectors.load_word2vec_format('/home/GloVe/glove_model_pdg.txt', binary=False)
    pool = Pool(4)
    pool.map(partial(joern_to_devign, word_vectors=word_vectors, out_path=out_path), dots)
    # for dot in dots:
    #    joern_to_devign(dot, word_vectors, out_path)

if __name__ == '__main__':
    main()

