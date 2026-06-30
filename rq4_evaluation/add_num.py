import os

def add_line_numbers_to_file(input_path, output_path):
    """
    Reads the source code and strictly adds the 'Line X: ' prefix to each line (including empty lines).
    """
    try:
        # Use utf-8 for reading; ignore occasional encoding errors to prevent interruption
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines, 1):
                # line from readlines() already includes \n at the end (except possibly the last line)
                # So we simply prepend the prefix; empty lines will become "Line X: \n"
                f.write(f"Line {i}: {line}")
                
    except Exception as e:
        print(f"❌ Error processing file {input_path}: {e}")

def process_directory(input_dir, output_dir):
    """
    Batch process all code files within a directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Assuming the code consists of c/cpp files; add or remove extensions as needed
    valid_extensions = ('.c', '.cpp', '.h', '.py', '.java')
    
    files_processed = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(valid_extensions):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)
            
            add_line_numbers_to_file(in_path, out_path)
            files_processed += 1
            
    print(f"✅ Batch processing complete! Added line numbers to {files_processed} code files.")

# ================= Execution Example =================
if __name__ == "__main__":
    # [Mode 1] Process a single file:
    # add_line_numbers_to_file("test.c", "test_with_lines.c")
    
    # [Mode 2] Batch process an entire directory:
    INPUT_FOLDER = "/home/1_vul"    # Replace with your source directory path
    OUTPUT_FOLDER = "/home/1_vul_num"   # Replace with your desired output directory path
    
    # Uncomment the line below to run batch processing as needed
    process_directory(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print("Script is ready. Please call Mode 1 or Mode 2 as needed!")