import os
import re
from clean_gadget import clean_gadget

def pro_one_file(filepath, output_dir):
    """
    Normalization logic for a single file.
    """
    file_name = os.path.basename(filepath)
    print(f"Processing: {filepath}")
    
    try:
        with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
            code = f.read()

        # --- Optimized comment handling ---
        # Remove // style comments
        code = re.sub(r'//.*', '', code)
        # Remove /* */ style comments, preserving newlines to maintain line numbering
        code = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group().count('\n'), code, flags=re.DOTALL)
        
        # Perform normalization
        org_lines = code.splitlines(keepends=True)
        nor_code_list = clean_gadget(org_lines)
        
        # Save the result
        save_path = os.path.join(output_dir, file_name)
        with open(save_path, "w", encoding='utf-8') as f:
            f.writelines(nor_code_list)
            
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

def normalize(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Check all files in the directory
    file_list = [f for f in os.listdir(input_path) if f.endswith('.c') or f.endswith('.cpp')]
    for _file in file_list:
        full_path = os.path.join(input_path, _file)
        pro_one_file(full_path, output_path)

if __name__ == '__main__':
    # Replace with your actual directory paths
    input_vul = "/0_src/0_novul"
    output_vul = "/1_norm/0_novul"
    
    normalize(input_vul, output_vul)
    print("All files processed successfully.")