import os
import json
import pickle
import re
import html
import networkx as nx
from glob import glob

# Common integer type keywords
INTEGER_TYPES = {
    'int', 'long', 'short', 'char', 'size_t', 'ssize_t', 
    'unsigned', 'signed', 'u8', 'u16', 'u32', 'u64', 
    'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uintptr_t', 'TAG_'
}

# Node types to ignore during processing
IGNORE_NODE_TYPES = {
    'IDENTIFIER', 'LITERAL', 'RETURN', 'METHOD', 'METHOD_RETURN', 
    'PARAM', 'BLOCK', 'UNKNOWN', 'FIELD_IDENTIFIER', 'CONTROL_STRUCTURE', 
    'JUMP_TARGET', 'MODIFIER', 'TYPE_DECL', 'TYPE_REF'
}

def identify_target_vars(node_dict):
    """Identify target variables from the '_4' field in the dictionary"""
    targets = {'pointers': set(), 'arrays': set(), 'integers': set()}
    items = node_dict.get('local', []) + node_dict.get('parameter', [])
    
    for item in items:
        if '_3' not in item or '_4' not in item: continue
        v_name, v_type = item['_3'], item['_4']
        
        if '*' in v_type:
            targets['pointers'].add(v_name)
        elif '[' in v_type and ']' in v_type:
            targets['arrays'].add(v_name)
        elif any(t in v_type for t in INTEGER_TYPES):
            targets['integers'].add(v_name)
    return targets

def find_slice_seeds(pdg, targets, sensitive_funcs):
    """Traverse PDG nodes and parse labels to find assignment points and APIs"""
    seeds = {} # Format: { "VAR_1": [node_id_1, node_id_2], "FUN_10": [node_id_3] }
    
    for node in pdg.nodes():
        raw_label = pdg.nodes[node].get('label', '')
        if not raw_label: continue
        
        # 1. Preprocess Label (Compatibility with newer Joern versions)
        label = html.unescape(raw_label.strip())
        if (label.startswith('<') and label.endswith('>')) or \
           (label.startswith('"') and label.endswith('"')):
            label = label[1:-1]
            
        if '<BR/>' not in label: continue
        
        parts = label.split('<BR/>')
        header = parts[0]
        actual_code = parts[-1].strip()
        op_type = header.split(',')[0].strip() # e.g., "<operator>.assignment" or "FUN_10"

        if op_type in IGNORE_NODE_TYPES:
            continue
        
        # 2. Look for sensitive APIs (Types without 'operator' are usually function calls)
        if 'operator' not in op_type:
            if op_type in sensitive_funcs:
                if op_type not in seeds: 
                    seeds[op_type] = []
                seeds[op_type].append(node)
        
        # 3. Look for variable assignment statements
        if 'assignment' in op_type:
            # Extract left-hand side (LHS)
            lhs = actual_code.split('=')[0].strip()
            # Extract base variable names (filter out *, ->, [], etc.)
            base_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lhs)
            
            for var in base_vars:
                is_target = (var in targets['pointers'] or 
                             var in targets['arrays'] or 
                             var in targets['integers'])
                if is_target:
                    if var not in seeds: seeds[var] = []
                    seeds[var].append(node)
                    
    return seeds

def compute_combined_slice(graph, start_nodes):
    """Merge forward and backward traversals"""
    slice_nodes = set()
    for start_node in start_nodes:
        if start_node not in graph: continue
        
        # Forward: Data/Control flow influenced by this node
        forward = nx.descendants(graph, start_node)
        forward.add(start_node)
        
        # Backward: Data/Control flow influencing this node
        backward = nx.ancestors(graph, start_node)
        backward.add(start_node)
        
        slice_nodes.update(forward)
        slice_nodes.update(backward)
        
    return slice_nodes

def process_slicing(pdg_dir, dict_dir, out_dir, sensitive_pkl):
    """Main process for slicing PDGs based on sensitive functions and variables"""
    with open(sensitive_pkl, "rb") as f:
        sensitive_funcs = pickle.load(f)
        if isinstance(sensitive_funcs, dict): sensitive_funcs = list(sensitive_funcs.keys())

    # Traverse all PDG dot files
    dot_files = glob(os.path.join(pdg_dir, "*.dot"))
    
    for dot_path in dot_files:
        try:
            # Parse filename (Assumes PDG naming: source_name.dot)
            dot_filename = os.path.basename(dot_path)
            # Extract source_name to match corresponding dict.json
            source_name = dot_filename.split('.dot')[0]
            
            # Compatibility: If corresponding dict is missing, skip variable slicing and only slice APIs
            dict_path = os.path.join(dict_dir, f"{source_name}_dict.json")
            targets = {'pointers': set(), 'arrays': set(), 'integers': set()}
            if os.path.exists(dict_path):
                with open(dict_path, 'r') as f:
                    node_dict = json.load(f)
                targets = identify_target_vars(node_dict)

            # Read PDG graph
            pdg = nx.nx_agraph.read_dot(dot_path)
            
            # Find all slicing starting points (multiple assignment points for the same variable are merged)
            slice_seeds = find_slice_seeds(pdg, targets, sensitive_funcs)
            if not slice_seeds: continue
            
            print(f"[+] Slicing {dot_filename} -> Found targets: {list(slice_seeds.keys())}")
            
            # Create a separate folder for the current PDG results
            graph_out_dir = os.path.join(out_dir, dot_filename.replace('.dot', ''))
            if not os.path.exists(graph_out_dir):
                os.makedirs(graph_out_dir)

            # Execute slicing
            for target_name, node_ids in slice_seeds.items():
                relevant_nodes = compute_combined_slice(pdg, node_ids)
                
                # Filter out slices that are too small to be valid
                if len(relevant_nodes) < 3: 
                    continue
                    
                subgraph = pdg.subgraph(relevant_nodes).copy()
                
                # Save format: PDG_Original_Name@@@Variable_or_API_Name.dot
                save_name = f"{dot_filename.replace('.dot', '')}@@@{target_name}.dot"
                save_path = os.path.join(graph_out_dir, save_name)
                
                nx.nx_agraph.write_dot(subgraph, save_path)
                
        except Exception as e:
            print(f"[Error] Processing {dot_path}: {e}")

if __name__ == '__main__':
    # Configuration paths
    PDG_DIR = "/home/Dataset/primevul/3_export/pdg_norm/0_novul"    # Directory containing PDG .dot files
    DICT_DIR = "/home/Dataset/primevul/5_node_dict/0_novul"             # Directory containing node_dict files
    OUTPUT_DIR = "/home/VGExplainer/slice/0_novul"                     # Root directory for saving slices
    SENSITIVE_PKL = "sensitive_func.pkl"            # Path to the sensitive functions pickle file
    
    process_slicing(PDG_DIR, DICT_DIR, OUTPUT_DIR, SENSITIVE_PKL)