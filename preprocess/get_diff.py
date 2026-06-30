import os
import difflib

def generate_unified_diff(vul_path, novul_path, out_diff_path):
    """
    Compares the pre-patch (vulnerable) and post-patch (non-vulnerable) code 
    to generate a standard Unified Diff.
    """
    # Read pre-patch code (vulnerable version -> corresponds to '-' in diff)
    with open(vul_path, 'r', encoding='utf-8', errors='ignore') as f:
        vul_lines = f.readlines()
        
    # Read post-patch code (fixed version -> corresponds to '+' in diff)
    with open(novul_path, 'r', encoding='utf-8', errors='ignore') as f:
        novul_lines = f.readlines()

    # Generate diff using difflib
    # n=3 retains 3 lines of context (standard diff format)
    diff = difflib.unified_diff(
        vul_lines, 
        novul_lines, 
        fromfile='a/vul_func.c', # Fake an a/ path for compatibility with standard diff parsers
        tofile='b/novul_func.c', # Fake a b/ path
        n=3
    )
    
    # Write to output file
    with open(out_diff_path, 'w', encoding='utf-8') as f:
        f.writelines(diff)

def batch_process_diffs(vul_dir, novul_dir, output_dir):
    """
    Batch process code pairs in the directories.
    Assumes that the filenames of vulnerable and fixed code are consistent 
    or can be mapped through specific rules.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through files in the vulnerable directory
    for filename in os.listdir(vul_dir):
        # Assuming a naming correspondence, e.g., both are 12345.c
        # If prefixes differ (e.g., 1_123.c and 0_123.c), perform string replacement here
        vul_file_path = os.path.join(vul_dir, filename)
        
        # Look for the corresponding fixed file
        # Example: Assume pre-patch is 1_xxx.c and post-patch is 0_xxx.c
        novul_filename = filename.replace('1_', '0_', 1) if filename.startswith('1_') else filename
        novul_file_path = os.path.join(novul_dir, novul_filename)
        
        if not os.path.exists(novul_file_path):
            print(f"Corresponding fixed file not found, skipping: {novul_filename}")
            continue
            
        # Set the output diff filename (e.g., 12345.diff)
        base_id = filename.replace('1_', '').split('.cpp')[0]
        out_diff_path = os.path.join(output_dir, f"{base_id}.diff")
        
        # Generate and save the diff
        generate_unified_diff(vul_file_path, novul_file_path, out_diff_path)
        
    print(f"All Diffs generated successfully, saved in: {output_dir}")

# ================= Execution Example =================
if __name__ == "__main__":
    # Please replace with your actual source code directory paths
    VULNERABLE_DIR = "/home/Dataset/primevul/0_src/1_vul"      # Pre-patch function code
    FIXED_DIR = "/home/Dataset/primevul/0_src/0_novul"          # Post-patch function code
    OUTPUT_DIFF_DIR = "/home/Dataset/primevul/0_src/diff"  # Location to save generated diffs
    
    batch_process_diffs(VULNERABLE_DIR, FIXED_DIR, OUTPUT_DIFF_DIR)