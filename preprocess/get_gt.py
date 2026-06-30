import os
import re
import networkx as nx
import json
from tqdm import tqdm


class GroundTruthGenerator:
    def __init__(self, diff_dir, dot_0_dir, dot_1_dir, out_gt_0, out_gt_1):
        self.diff_dir = diff_dir
        self.dot_0_dir = dot_0_dir # DOT directory for fixed version (Class 0)
        self.dot_1_dir = dot_1_dir # DOT directory for vulnerable version (Class 1)
        self.out_gt_0 = out_gt_0
        self.out_gt_1 = out_gt_1
        
        os.makedirs(out_gt_0, exist_ok=True)
        os.makedirs(out_gt_1, exist_ok=True)
        
        # Filter list for common keywords to prevent matching if, return, etc. as variables
        self.keywords = {
            "if", "else", "for", "while", "do", "switch", "case", "return", "sizeof",
            "int", "char", "float", "double", "void", "struct", "class", "goto", "break",
            "continue", "unsigned", "long", "short", "static", "const", "size_t", "bool",
            "true", "false", "NULL", "nullptr", "auto", "new", "delete"
        }

    def parse_diff(self, diff_path):
        """
        Intelligently parse diff files:
        1. Extract line numbers for pre-patch (deleted) and post-patch (added) versions.
        2. For purely additive patches: extract new variables and trace them downward 
           in context to pinpoint the trigger point.
        """
        added_lines = []
        deleted_lines = []
        anchor_lines = []
        
        with open(diff_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        old_ln = 0
        new_ln = 0
        
        # State machine for intelligent anchor tracking
        pending_vars = set()
        fallback_anchor = None
        looking_for_anchor = False

        def resolve_anchor():
            """Finish current search; if no variable match is found, use the fallback line (first line directly below)"""
            nonlocal looking_for_anchor, fallback_anchor, pending_vars
            if looking_for_anchor and fallback_anchor is not None:
                anchor_lines.append(fallback_anchor)
            looking_for_anchor = False
            fallback_anchor = None
            pending_vars.clear()

        for line in lines:
            hunk_match = re.match(r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if hunk_match:
                resolve_anchor() # Settle before leaving the previous hunk
                old_ln = int(hunk_match.group(1))
                new_ln = int(hunk_match.group(2))
                continue
                
            if line.startswith('---') or line.startswith('+++'):
                continue
                
            if line.startswith('-'):
                resolve_anchor()
                deleted_lines.append(old_ln)
                old_ln += 1
                
            elif line.startswith('+'):
                # If a new '+' block is encountered while still searching, settle the previous block first
                if looking_for_anchor and fallback_anchor is not None:
                    resolve_anchor()
                    
                added_lines.append(new_ln)
                new_ln += 1
                looking_for_anchor = True
                
                # Extract all words from current added line excluding keywords
                words = set(re.findall(r'[a-zA-Z_]\w*', line[1:]))
                pending_vars.update(words - self.keywords)
                
            elif line.startswith(' '):
                if looking_for_anchor:
                    # Record the first line directly below a '+' block as a fallback
                    if fallback_anchor is None:
                        fallback_anchor = old_ln 
                    
                    # Check if current context line uses the newly added variables
                    words = set(re.findall(r'[a-zA-Z_]\w*', line[1:]))
                    if pending_vars.intersection(words):
                        # Match found! Precise hit on the trigger point
                        anchor_lines.append(old_ln)
                        looking_for_anchor = False # End search
                        fallback_anchor = None
                        pending_vars.clear()
                        
                old_ln += 1
                new_ln += 1
                
        resolve_anchor() # Final settlement at end of file
        return added_lines, deleted_lines, anchor_lines

    def extract_lines_from_dot(self, dot_path):
        """Extract valid line numbers and node mappings from a DOT file"""
        valid_lines = set()
        line2nodes = {}
        try:
            pdg = nx.nx_agraph.read_dot(dot_path)
        except Exception as e:
            return valid_lines, line2nodes, None

        line_regex = re.compile(r',\s*(\d+)\s*<BR/>')
        for node in pdg.nodes():
            label = pdg.nodes[node].get('label', '')
            match = line_regex.search(label)
            if match:
                lineno = int(match.group(1))
                valid_lines.add(lineno)
                if lineno not in line2nodes:
                    line2nodes[lineno] = []
                line2nodes[lineno].append(node)
                
        return valid_lines, line2nodes, pdg

    def get_pdg_slice(self, pdg, start_nodes, k_hop=2):
        """Calculate PDG slice based on target starting nodes"""
        reachable_nodes = set(start_nodes)
        
        for node in start_nodes:
            label = pdg.nodes[node].get('label', '')
            is_control_flow = bool(re.search(r'<BR/>\s*(if|while|for|switch)\b', label))
            
            # 1. Full backward slice
            reachable_nodes.update(nx.ancestors(pdg, node))
            
            if is_control_flow:
                # 2. Control flow: Full forward slice
                reachable_nodes.update(nx.descendants(pdg, node))
            else:
                # 3. Regular statements: K-hop forward slice
                downstream_nodes = nx.single_source_shortest_path_length(pdg, node, cutoff=k_hop)
                reachable_nodes.update(downstream_nodes.keys())
            
        slice_lines = set()
        line_regex = re.compile(r',\s*(\d+)\s*<BR/>')
        for node in reachable_nodes:
            label = pdg.nodes[node].get('label', '')
            match = line_regex.search(label)
            if match:
                slice_lines.add(int(match.group(1)))
                
        return sorted(list(slice_lines))

    def process_all(self):
        """Process all diff files to extract ground truth for Class 0 and Class 1"""
        diff_files = [f for f in os.listdir(self.diff_dir) if f.endswith('.diff')]
        
        for diff_file in tqdm(diff_files, desc="Processing Ground Truth"):
            base_name = diff_file.replace('.diff', '')
            diff_path = os.path.join(self.diff_dir, diff_file)
            
            dot_0_path = os.path.join(self.dot_0_dir, f"0_{base_name}.dot")
            dot_1_path = os.path.join(self.dot_1_dir, f"1_{base_name}.dot")
            
            added_lines, deleted_lines, anchor_lines = self.parse_diff(diff_path)
            
            # --- Task 1: Class 0 (Post-patch / Fixed) ---
            if os.path.exists(dot_0_path):
                valid_lines_0, _, _ = self.extract_lines_from_dot(dot_0_path)
                gt_0 = [ln for ln in added_lines if ln in valid_lines_0]
                
                with open(os.path.join(self.out_gt_0, f"{base_name}.json"), 'w') as f:
                    json.dump({"ground_truth": gt_0, "raw_added": added_lines}, f)
                    
            # --- Task 2: Class 1 (Pre-patch / Vulnerable) ---
            if os.path.exists(dot_1_path):
                valid_lines_1, line2nodes_1, pdg_1 = self.extract_lines_from_dot(dot_1_path)
                
                target_lines = [ln for ln in deleted_lines if ln in valid_lines_1]
                
                # If no deleted lines (additive patch), use intelligently calculated anchor lines
                if not target_lines:
                    target_lines = [ln for ln in anchor_lines if ln in valid_lines_1]
                    
                gt_1 = []
                if target_lines and pdg_1:
                    start_nodes = []
                    for ln in target_lines:
                        start_nodes.extend(line2nodes_1.get(ln, []))
                        
                    if start_nodes:
                        gt_1 = self.get_pdg_slice(pdg_1, start_nodes, k_hop=2)
                
                with open(os.path.join(self.out_gt_1, f"{base_name}.json"), 'w') as f:
                    json.dump({"ground_truth": gt_1, "slice_starts": target_lines}, f)

# Execution Example
if __name__ == "__main__":
    generator = GroundTruthGenerator(
        diff_dir="/home/Dataset/primevul/0_src/diff",
        dot_0_dir="/home/Dataset/primevul/3_export/pdg_norm/0_novul",  # Fixed graph path
        dot_1_dir="/home/Dataset/primevul/3_export/pdg_norm/1_vul",    # Vulnerable graph path
        out_gt_0="/home/Dataset/primevul/6_ground_truth/gt_0",
        out_gt_1="/home/Dataset/primevul/6_ground_truth/gt_1"
    )
    generator.process_all()
    print("Ground Truth extraction complete!")