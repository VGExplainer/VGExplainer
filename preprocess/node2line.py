import os, sys
import glob
import shutil
import networkx as nx
import json
import glob
from tqdm import tqdm

if __name__ == "__main__":
    src_path='raw_results(nodes)_path'
    dict_path='node2line_dict_path'
    #lineinfo='/home/mytest/nvd/nvd_vul_lineinfo.json'
    dest_path='line_results_path'
    #with open(lineinfo,'r') as lp:
    #    vul_line=json.load(lp)
    res_files=glob.glob(src_path+'/*')
    for res_file in tqdm(res_files):
        func_name = res_file.split('/')[-1]
        dict_name = func_name+'.json'
        slice_res = glob.glob(res_file+'/*')
        try:
            with open(dict_path+dict_name,'r') as rf2:
                all_nodes=json.load(rf2)
                rf2.close()
        except:
            continue
        for slice_item in slice_res:
            res_lines=[]
            slice_name=slice_item.split('/')[-1]
            if not os.path.exists(dest_path+func_name):
                os.mkdir(dest_path+func_name)
            with open(slice_item,'r') as rf:
                key_nodes=json.load(rf)
                rf.close()
            for key_node in key_nodes:
                for node_x in all_nodes:
                    if node_x['node'] == key_node:
                        res_lines.append(int(node_x['line_num']))
            res_lines=list(set(res_lines))
            dest_path2 = dest_path+func_name+'/'
            with open(dest_path2+slice_name,'w') as wf:
                json.dump(res_lines,wf)